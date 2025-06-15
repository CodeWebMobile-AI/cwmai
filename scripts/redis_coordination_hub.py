"""
Redis Coordination Hub

Advanced coordination system using Redis Pub/Sub, distributed locking, and 
coordination patterns for distributed worker intelligence management.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Callable, Set, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

from redis_integration import get_redis_client, RedisPubSubManager, RedisLocksManager
from redis_lockfree_adapter import create_lockfree_state_manager
from redis_intelligence_hub import IntelligenceEvent, EventType, EventPriority


class CoordinationMessageType(Enum):
    """Types of coordination messages."""
    LEADER_ELECTION = "leader_election"
    HEARTBEAT = "heartbeat"
    TASK_BROADCAST = "task_broadcast"
    WORKER_ANNOUNCEMENT = "worker_announcement"
    SYSTEM_ALERT = "system_alert"
    COORDINATION_REQUEST = "coordination_request"
    CONSENSUS_PROPOSAL = "consensus_proposal"
    LOAD_BALANCE = "load_balance"
    FAILOVER = "failover"
    SCALING_EVENT = "scaling_event"


class CoordinationPriority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class CoordinationMessage:
    """Coordination message structure."""
    message_id: str
    message_type: CoordinationMessageType
    sender_id: str
    target: Optional[str]  # Specific target or None for broadcast
    priority: CoordinationPriority
    timestamp: datetime
    ttl_seconds: int
    data: Dict[str, Any]
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for Redis."""
        return {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'sender_id': self.sender_id,
            'target': self.target,
            'priority': self.priority.value,
            'timestamp': self.timestamp.isoformat(),
            'ttl_seconds': self.ttl_seconds,
            'data': json.dumps(self.data),
            'correlation_id': self.correlation_id,
            'reply_to': self.reply_to
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CoordinationMessage':
        """Create message from dictionary."""
        return cls(
            message_id=data['message_id'],
            message_type=CoordinationMessageType(data['message_type']),
            sender_id=data['sender_id'],
            target=data.get('target'),
            priority=CoordinationPriority(data['priority']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            ttl_seconds=data['ttl_seconds'],
            data=json.loads(data['data']),
            correlation_id=data.get('correlation_id'),
            reply_to=data.get('reply_to')
        )


class LeaderElection:
    """Distributed leader election using Redis."""
    
    def __init__(self, 
                 node_id: str, 
                 election_key: str = "leader_election",
                 lease_duration: int = 30,
                 election_timeout: int = 10):
        """Initialize leader election.
        
        Args:
            node_id: Unique node identifier
            election_key: Redis key for leader election
            lease_duration: Leader lease duration in seconds
            election_timeout: Election timeout in seconds
        """
        self.node_id = node_id
        self.election_key = election_key
        self.lease_duration = lease_duration
        self.election_timeout = election_timeout
        
        self.logger = logging.getLogger(f"{__name__}.LeaderElection")
        self.redis_client = redis_client  # shared client if provided, else will fetch below
        self.locks_manager: Optional[RedisLocksManager] = None
        
        self.is_leader = False
        self.current_leader = None
        self.election_in_progress = False
        self._leader_task: Optional[asyncio.Task] = None
        self._shutdown = False
    
    async def initialize(self):
        """Initialize leader election components."""
        # reuse shared RedisClient, do not open new pool
        self.locks_manager = RedisLocksManager(self.redis_client)
    
    async def start_election(self) -> bool:
        """Start leader election process."""
        if self.election_in_progress:
            return False
        
        try:
            self.election_in_progress = True
            self.logger.info(f"Node {self.node_id} starting leader election")
            
            # Try to acquire leader lock
            leader_lock = f"leader:{self.election_key}"
            
            acquired = await self.locks_manager.acquire_lock(
                leader_lock,
                self.node_id,
                self.lease_duration
            )
            
            if acquired:
                self.is_leader = True
                self.current_leader = self.node_id
                self.logger.info(f"Node {self.node_id} elected as leader")
                
                # Start leader maintenance task
                self._leader_task = asyncio.create_task(self._maintain_leadership())
                return True
            else:
                # Check current leader
                current_leader = await self.redis_client.get(leader_lock)
                if current_leader:
                    self.current_leader = current_leader.decode() if isinstance(current_leader, bytes) else current_leader
                    self.logger.info(f"Node {self.current_leader} is current leader")
                
                return False
                
        except Exception as e:
            self.logger.error(f"Error in leader election: {e}")
            return False
        finally:
            self.election_in_progress = False
    
    async def _maintain_leadership(self):
        """Maintain leadership by renewing lease."""
        leader_lock = f"leader:{self.election_key}"
        
        while not self._shutdown and self.is_leader:
            try:
                # Renew leadership lease
                renewed = await self.locks_manager.extend_lock(
                    leader_lock,
                    self.node_id,
                    self.lease_duration
                )
                
                if not renewed:
                    self.logger.warning(f"Node {self.node_id} lost leadership")
                    self.is_leader = False
                    break
                
                # Sleep for half the lease duration before renewing
                await asyncio.sleep(self.lease_duration / 2)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error maintaining leadership: {e}")
                self.is_leader = False
                break
    
    async def step_down(self):
        """Voluntarily step down from leadership."""
        if self.is_leader:
            leader_lock = f"leader:{self.election_key}"
            await self.locks_manager.release_lock(leader_lock, self.node_id)
            self.is_leader = False
            self.current_leader = None
            
            if self._leader_task:
                self._leader_task.cancel()
            
            self.logger.info(f"Node {self.node_id} stepped down from leadership")
    
    async def get_leader(self) -> Optional[str]:
        """Get current leader."""
        if self.is_leader:
            return self.node_id
        
        leader_lock = f"leader:{self.election_key}"
        leader = await self.redis_client.get(leader_lock)
        return leader.decode() if isinstance(leader, bytes) else leader
    
    async def shutdown(self):
        """Shutdown leader election."""
        self._shutdown = True
        if self.is_leader:
            await self.step_down()


class ConsensusManager:
    """Distributed consensus mechanism using Redis."""
    
    def __init__(self, node_id: str, quorum_size: int = None):
        """Initialize consensus manager.
        
        Args:
            node_id: Unique node identifier
            quorum_size: Minimum nodes needed for consensus (default: majority)
        """
        self.node_id = node_id
        self.quorum_size = quorum_size
        
        self.logger = logging.getLogger(f"{__name__}.ConsensusManager")
        self.redis_client = None
        self.state_manager = None
        
        self._active_proposals: Dict[str, Dict[str, Any]] = {}
        self._shutdown = False
    
    async def initialize(self):
        """Initialize consensus components."""
        # reuse shared RedisClient, do not open new pool
        self.state_manager = create_lockfree_state_manager(f"coordination_hub_{self.hub_id}")
        await self.state_manager.initialize()
        
        # Determine quorum size if not specified
        if self.quorum_size is None:
            # Get number of active nodes (simplified)
            self.quorum_size = 3  # Default to 3 for majority in 5-node cluster
    
    async def propose(self, 
                     proposal_id: str, 
                     proposal_data: Dict[str, Any],
                     timeout_seconds: int = 30) -> bool:
        """Propose a value for consensus.
        
        Args:
            proposal_id: Unique proposal identifier
            proposal_data: Data to reach consensus on
            timeout_seconds: Timeout for consensus
            
        Returns:
            True if consensus achieved
        """
        try:
            self.logger.info(f"Node {self.node_id} proposing {proposal_id}")
            
            # Create proposal
            proposal = {
                'proposal_id': proposal_id,
                'proposer': self.node_id,
                'data': proposal_data,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'timeout': timeout_seconds,
                'votes': {self.node_id: 'yes'},  # Self-vote
                'status': 'active'
            }
            
            # Store proposal
            proposal_key = f"consensus:proposals:{proposal_id}"
            await self.redis_client.setex(
                proposal_key,
                timeout_seconds,
                json.dumps(proposal)
            )
            
            # Track locally
            self._active_proposals[proposal_id] = proposal
            
            # Broadcast proposal
            await self._broadcast_proposal(proposal)
            
            # Wait for consensus or timeout
            return await self._wait_for_consensus(proposal_id, timeout_seconds)
            
        except Exception as e:
            self.logger.error(f"Error in proposal {proposal_id}: {e}")
            return False
    
    async def vote(self, proposal_id: str, vote: str) -> bool:
        """Vote on a proposal.
        
        Args:
            proposal_id: Proposal to vote on
            vote: Vote value ('yes', 'no', 'abstain')
            
        Returns:
            True if vote recorded
        """
        try:
            proposal_key = f"consensus:proposals:{proposal_id}"
            proposal_data = await self.redis_client.get(proposal_key)
            
            if not proposal_data:
                self.logger.warning(f"Proposal {proposal_id} not found")
                return False
            
            proposal = json.loads(proposal_data)
            
            # Record vote
            proposal['votes'][self.node_id] = vote
            
            # Update proposal
            await self.redis_client.setex(
                proposal_key,
                proposal['timeout'],
                json.dumps(proposal)
            )
            
            self.logger.info(f"Node {self.node_id} voted {vote} on {proposal_id}")
            
            # Check if consensus reached
            await self._check_consensus(proposal)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error voting on {proposal_id}: {e}")
            return False
    
    async def _broadcast_proposal(self, proposal: Dict[str, Any]):
        """Broadcast proposal to all nodes."""
        # This would integrate with the coordination hub's pub/sub system
        # For now, we'll store in a broadcast channel
        broadcast_key = f"consensus:broadcast:{proposal['proposal_id']}"
        await self.redis_client.setex(
            broadcast_key,
            proposal['timeout'],
            json.dumps(proposal)
        )
    
    async def _wait_for_consensus(self, proposal_id: str, timeout_seconds: int) -> bool:
        """Wait for consensus on proposal."""
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            proposal_key = f"consensus:proposals:{proposal_id}"
            proposal_data = await self.redis_client.get(proposal_key)
            
            if not proposal_data:
                return False
            
            proposal = json.loads(proposal_data)
            
            if proposal['status'] == 'approved':
                return True
            elif proposal['status'] == 'rejected':
                return False
            
            await asyncio.sleep(1)  # Check every second
        
        # Timeout - mark as failed
        await self._mark_proposal_failed(proposal_id, "timeout")
        return False
    
    async def _check_consensus(self, proposal: Dict[str, Any]):
        """Check if consensus has been reached."""
        votes = proposal['votes']
        yes_votes = sum(1 for vote in votes.values() if vote == 'yes')
        no_votes = sum(1 for vote in votes.values() if vote == 'no')
        
        if yes_votes >= self.quorum_size:
            proposal['status'] = 'approved'
            await self._apply_consensus_decision(proposal)
            self.logger.info(f"Consensus reached: {proposal['proposal_id']} approved")
        
        elif no_votes >= self.quorum_size:
            proposal['status'] = 'rejected'
            self.logger.info(f"Consensus reached: {proposal['proposal_id']} rejected")
    
    async def _apply_consensus_decision(self, proposal: Dict[str, Any]):
        """Apply the consensus decision."""
        # Store the consensus result
        result_key = f"consensus:results:{proposal['proposal_id']}"
        await self.state_manager.update(
            result_key,
            {
                'proposal_id': proposal['proposal_id'],
                'decision': 'approved',
                'data': proposal['data'],
                'votes': proposal['votes'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            distributed=True
        )
    
    async def _mark_proposal_failed(self, proposal_id: str, reason: str):
        """Mark proposal as failed."""
        proposal_key = f"consensus:proposals:{proposal_id}"
        proposal_data = await self.redis_client.get(proposal_key)
        
        if proposal_data:
            proposal = json.loads(proposal_data)
            proposal['status'] = 'failed'
            proposal['failure_reason'] = reason
            
            await self.redis_client.setex(
                proposal_key,
                60,  # Keep failed proposals for 1 minute
                json.dumps(proposal)
            )


class RedisCoordinationHub:
    """Advanced Redis-based coordination hub with Pub/Sub, locking, and consensus."""
    
    def __init__(self,
                 hub_id: str = None,
                 enable_leader_election: bool = True,
                 enable_consensus: bool = True,
                 enable_load_balancing: bool = True):
        """Initialize Redis coordination hub.
        
        Args:
            hub_id: Unique hub identifier
            enable_leader_election: Enable leader election
            enable_consensus: Enable consensus mechanisms
            enable_load_balancing: Enable automatic load balancing
        """
        self.hub_id = hub_id or f"coord_hub_{uuid.uuid4().hex[:8]}"
        self.enable_leader_election = enable_leader_election
        self.enable_consensus = enable_consensus
        self.enable_load_balancing = enable_load_balancing
        
        self.logger = logging.getLogger(f"{__name__}.RedisCoordinationHub")
        
        # Redis components
        self.redis_client = None
        self.pubsub_manager: Optional[RedisPubSubManager] = None
        self.locks_manager: Optional[RedisLocksManager] = None
        self.state_manager = None
        
        # Coordination components
        self.leader_election: Optional[LeaderElection] = None
        self.consensus_manager: Optional[ConsensusManager] = None
        
        # Pub/Sub channels
        self.channels = {
            'coordination': f'coord:{self.hub_id}',
            'broadcast': f'broadcast:{self.hub_id}',
            'leadership': f'leadership:{self.hub_id}',
            'consensus': f'consensus:{self.hub_id}',
            'alerts': f'alerts:{self.hub_id}',
            'load_balance': f'loadbal:{self.hub_id}'
        }
        
        # Message handlers
        self._message_handlers: Dict[CoordinationMessageType, List[Callable]] = {}
        self._active_subscriptions: Set[str] = set()
        
        # Coordination state
        self._connected_nodes: Dict[str, Dict[str, Any]] = {}
        self._coordination_tasks: List[asyncio.Task] = []
        self._shutdown = False
        
        # Performance metrics
        self._metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'leader_elections': 0,
            'consensus_proposals': 0,
            'locks_acquired': 0,
            'coordination_events': 0
        }
    
    async def initialize(self):
        """Initialize coordination hub components."""
        try:
            self.logger.info(f"Initializing Redis Coordination Hub: {self.hub_id}")
            
            # Initialize Redis components
            # reuse shared RedisClient, do not open new pool
            self.pubsub_manager = RedisPubSubManager(self.redis_client)
            self.locks_manager = RedisLocksManager(self.redis_client)
            self.state_manager = create_lockfree_state_manager(f"coordination_hub_{self.hub_id}")
            await self.state_manager.initialize()
            
            # Initialize coordination components
            if self.enable_leader_election:
                self.leader_election = LeaderElection(self.hub_id)
                await self.leader_election.initialize()
            
            if self.enable_consensus:
                self.consensus_manager = ConsensusManager(self.hub_id)
                await self.consensus_manager.initialize()
            
            # Set up pub/sub subscriptions
            await self._setup_pubsub_subscriptions()
            
            # Start coordination tasks
            await self._start_coordination_tasks()
            
            # Register hub
            await self._register_coordination_hub()
            
            self.logger.info(f"Coordination Hub {self.hub_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing Coordination Hub: {e}")
            raise
    
    async def _setup_pubsub_subscriptions(self):
        """Set up Pub/Sub subscriptions for coordination."""
        try:
            # Subscribe to coordination channels
            for channel_name, channel_key in self.channels.items():
                await self.pubsub_manager.subscribe(channel_key, self._handle_coordination_message)
                self._active_subscriptions.add(channel_key)
            
            # Start pub/sub processing
            await self.pubsub_manager.start_processing()
            
            self.logger.info(f"Subscribed to {len(self._active_subscriptions)} coordination channels")
            
        except Exception as e:
            self.logger.error(f"Error setting up pub/sub subscriptions: {e}")
            raise
    
    async def _start_coordination_tasks(self):
        """Start background coordination tasks."""
        try:
            # Node heartbeat task
            heartbeat_task = asyncio.create_task(self._heartbeat_sender())
            self._coordination_tasks.append(heartbeat_task)
            
            # Leader election monitoring
            if self.enable_leader_election:
                election_task = asyncio.create_task(self._leader_election_monitor())
                self._coordination_tasks.append(election_task)
            
            # Load balancing task
            if self.enable_load_balancing:
                loadbal_task = asyncio.create_task(self._load_balancing_monitor())
                self._coordination_tasks.append(loadbal_task)
            
            # Node discovery task
            discovery_task = asyncio.create_task(self._node_discovery())
            self._coordination_tasks.append(discovery_task)
            
            # Cleanup task
            cleanup_task = asyncio.create_task(self._cleanup_expired_data())
            self._coordination_tasks.append(cleanup_task)
            
            self.logger.info(f"Started {len(self._coordination_tasks)} coordination tasks")
            
        except Exception as e:
            self.logger.error(f"Error starting coordination tasks: {e}")
    
    async def send_coordination_message(self, 
                                      message: CoordinationMessage,
                                      channel: str = None) -> bool:
        """Send coordination message via Pub/Sub.
        
        Args:
            message: Coordination message to send
            channel: Specific channel (defaults to coordination channel)
            
        Returns:
            True if message sent successfully
        """
        try:
            target_channel = channel or self.channels['coordination']
            
            # Add message metadata
            message_data = message.to_dict()
            message_data['sent_by_hub'] = self.hub_id
            message_data['sent_at'] = datetime.now(timezone.utc).isoformat()
            
            # Publish message
            success = await self.pubsub_manager.publish(target_channel, message_data)
            
            if success:
                self._metrics['messages_sent'] += 1
                self.logger.debug(f"Sent {message.message_type.value} message to {target_channel}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending coordination message: {e}")
            return False
    
    async def broadcast_message(self, 
                              message_type: CoordinationMessageType,
                              data: Dict[str, Any],
                              priority: CoordinationPriority = CoordinationPriority.NORMAL,
                              ttl_seconds: int = 60) -> bool:
        """Broadcast message to all nodes.
        
        Args:
            message_type: Type of message
            data: Message data
            priority: Message priority
            ttl_seconds: Message TTL
            
        Returns:
            True if broadcast successful
        """
        message = CoordinationMessage(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            sender_id=self.hub_id,
            target=None,  # Broadcast
            priority=priority,
            timestamp=datetime.now(timezone.utc),
            ttl_seconds=ttl_seconds,
            data=data
        )
        
        return await self.send_coordination_message(message, self.channels['broadcast'])
    
    async def send_targeted_message(self,
                                  target_node: str,
                                  message_type: CoordinationMessageType,
                                  data: Dict[str, Any],
                                  priority: CoordinationPriority = CoordinationPriority.NORMAL) -> bool:
        """Send message to specific node.
        
        Args:
            target_node: Target node ID
            message_type: Type of message
            data: Message data
            priority: Message priority
            
        Returns:
            True if message sent
        """
        message = CoordinationMessage(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            sender_id=self.hub_id,
            target=target_node,
            priority=priority,
            timestamp=datetime.now(timezone.utc),
            ttl_seconds=300,  # 5 minutes
            data=data
        )
        
        # Send to targeted channel
        target_channel = f"direct:{target_node}"
        return await self.send_coordination_message(message, target_channel)
    
    async def _handle_coordination_message(self, channel: str, message_data: Dict[str, Any]):
        """Handle incoming coordination message."""
        try:
            message = CoordinationMessage.from_dict(message_data)
            
            # Check if message is for this node
            if message.target and message.target != self.hub_id:
                return
            
            self._metrics['messages_received'] += 1
            self._metrics['coordination_events'] += 1
            
            # Handle message based on type
            await self._route_coordination_message(message)
            
            # Call registered handlers
            handlers = self._message_handlers.get(message.message_type, [])
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)
                except Exception as e:
                    self.logger.error(f"Error in message handler: {e}")
            
        except Exception as e:
            self.logger.error(f"Error handling coordination message: {e}")
    
    async def _route_coordination_message(self, message: CoordinationMessage):
        """Route coordination message to appropriate handler."""
        try:
            if message.message_type == CoordinationMessageType.HEARTBEAT:
                await self._handle_heartbeat_message(message)
            
            elif message.message_type == CoordinationMessageType.LEADER_ELECTION:
                await self._handle_leader_election_message(message)
            
            elif message.message_type == CoordinationMessageType.CONSENSUS_PROPOSAL:
                await self._handle_consensus_message(message)
            
            elif message.message_type == CoordinationMessageType.WORKER_ANNOUNCEMENT:
                await self._handle_worker_announcement(message)
            
            elif message.message_type == CoordinationMessageType.LOAD_BALANCE:
                await self._handle_load_balance_message(message)
            
            elif message.message_type == CoordinationMessageType.SYSTEM_ALERT:
                await self._handle_system_alert(message)
            
            elif message.message_type == CoordinationMessageType.FAILOVER:
                await self._handle_failover_message(message)
            
        except Exception as e:
            self.logger.error(f"Error routing coordination message: {e}")
    
    async def _handle_heartbeat_message(self, message: CoordinationMessage):
        """Handle heartbeat message."""
        sender_id = message.sender_id
        heartbeat_data = message.data
        
        # Update node registry
        self._connected_nodes[sender_id] = {
            'last_heartbeat': message.timestamp.isoformat(),
            'status': heartbeat_data.get('status', 'unknown'),
            'load': heartbeat_data.get('load', 0.0),
            'capabilities': heartbeat_data.get('capabilities', []),
            'metadata': heartbeat_data.get('metadata', {})
        }
        
        # Store in distributed state
        await self.state_manager.update(
            f"coordination.nodes.{sender_id}",
            self._connected_nodes[sender_id],
            distributed=True
        )
    
    async def _heartbeat_sender(self):
        """Send periodic heartbeat messages."""
        while not self._shutdown:
            try:
                # Get current node status
                node_status = await self._get_node_status()
                
                # Send heartbeat
                await self.broadcast_message(
                    CoordinationMessageType.HEARTBEAT,
                    node_status,
                    CoordinationPriority.LOW,
                    ttl_seconds=120
                )
                
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat sender: {e}")
                await asyncio.sleep(10)
    
    async def _get_node_status(self) -> Dict[str, Any]:
        """Get current node status for heartbeat."""
        return {
            'status': 'active',
            'load': 0.5,  # This would be calculated from actual metrics
            'capabilities': ['coordination', 'pubsub', 'locking'],
            'is_leader': self.leader_election.is_leader if self.leader_election else False,
            'metrics': self._metrics.copy(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def _leader_election_monitor(self):
        """Monitor and manage leader election."""
        while not self._shutdown:
            try:
                if not self.leader_election.is_leader and not self.leader_election.election_in_progress:
                    # Try to become leader if no current leader
                    current_leader = await self.leader_election.get_leader()
                    
                    if not current_leader:
                        success = await self.leader_election.start_election()
                        if success:
                            self._metrics['leader_elections'] += 1
                            
                            # Announce leadership
                            await self.broadcast_message(
                                CoordinationMessageType.LEADER_ELECTION,
                                {
                                    'event': 'leader_elected',
                                    'leader_id': self.hub_id,
                                    'election_time': datetime.now(timezone.utc).isoformat()
                                },
                                CoordinationPriority.HIGH
                            )
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in leader election monitor: {e}")
                await asyncio.sleep(30)
    
    def register_message_handler(self, 
                                message_type: CoordinationMessageType, 
                                handler: Callable):
        """Register handler for coordination message type.
        
        Args:
            message_type: Message type to handle
            handler: Handler function (sync or async)
        """
        if message_type not in self._message_handlers:
            self._message_handlers[message_type] = []
        
        self._message_handlers[message_type].append(handler)
        self.logger.info(f"Registered handler for {message_type.value}")
    
    async def acquire_distributed_lock(self, 
                                     lock_name: str, 
                                     timeout_seconds: int = 30,
                                     auto_release: bool = True) -> Optional[str]:
        """Acquire distributed lock.
        
        Args:
            lock_name: Name of lock to acquire
            timeout_seconds: Lock timeout
            auto_release: Automatically release lock
            
        Returns:
            Lock token if acquired, None otherwise
        """
        try:
            success = await self.locks_manager.acquire_lock(
                lock_name,
                self.hub_id,
                timeout_seconds
            )
            
            if success:
                self._metrics['locks_acquired'] += 1
                self.logger.debug(f"Acquired lock: {lock_name}")
                return f"{lock_name}:{self.hub_id}"
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error acquiring lock {lock_name}: {e}")
            return None
    
    async def release_distributed_lock(self, lock_name: str) -> bool:
        """Release distributed lock.
        
        Args:
            lock_name: Name of lock to release
            
        Returns:
            True if released successfully
        """
        try:
            return await self.locks_manager.release_lock(lock_name, self.hub_id)
        except Exception as e:
            self.logger.error(f"Error releasing lock {lock_name}: {e}")
            return False
    
    async def propose_consensus(self, 
                              proposal_id: str, 
                              proposal_data: Dict[str, Any]) -> bool:
        """Propose consensus decision.
        
        Args:
            proposal_id: Unique proposal identifier
            proposal_data: Data for consensus
            
        Returns:
            True if consensus achieved
        """
        if not self.consensus_manager:
            return False
        
        try:
            self._metrics['consensus_proposals'] += 1
            
            # Broadcast consensus proposal
            await self.broadcast_message(
                CoordinationMessageType.CONSENSUS_PROPOSAL,
                {
                    'proposal_id': proposal_id,
                    'proposal_data': proposal_data,
                    'proposer': self.hub_id
                },
                CoordinationPriority.HIGH
            )
            
            # Start consensus process
            return await self.consensus_manager.propose(proposal_id, proposal_data)
            
        except Exception as e:
            self.logger.error(f"Error in consensus proposal: {e}")
            return False
    
    async def get_coordination_status(self) -> Dict[str, Any]:
        """Get comprehensive coordination status."""
        return {
            'hub_id': self.hub_id,
            'is_leader': self.leader_election.is_leader if self.leader_election else False,
            'current_leader': await self.leader_election.get_leader() if self.leader_election else None,
            'connected_nodes': len(self._connected_nodes),
            'active_subscriptions': len(self._active_subscriptions),
            'coordination_channels': list(self.channels.keys()),
            'message_handlers': {mt.value: len(handlers) for mt, handlers in self._message_handlers.items()},
            'metrics': self._metrics.copy(),
            'capabilities': {
                'leader_election': self.enable_leader_election,
                'consensus': self.enable_consensus,
                'load_balancing': self.enable_load_balancing
            }
        }
    
    async def _register_coordination_hub(self):
        """Register coordination hub in distributed registry."""
        hub_data = {
            'hub_id': self.hub_id,
            'start_time': datetime.now(timezone.utc).isoformat(),
            'status': 'active',
            'channels': list(self.channels.values()),
            'capabilities': {
                'leader_election': self.enable_leader_election,
                'consensus': self.enable_consensus,
                'load_balancing': self.enable_load_balancing
            }
        }
        
        await self.state_manager.update(
            f"coordination.hubs.{self.hub_id}",
            hub_data,
            distributed=True
        )
    
    async def shutdown(self):
        """Shutdown coordination hub."""
        self.logger.info(f"Shutting down Coordination Hub: {self.hub_id}")
        self._shutdown = True
        
        # Step down from leadership
        if self.leader_election and self.leader_election.is_leader:
            await self.leader_election.step_down()
        
        # Stop coordination tasks
        for task in self._coordination_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop pub/sub processing
        if self.pubsub_manager:
            await self.pubsub_manager.stop_processing()
        
        # Update hub status
        await self.state_manager.update(
            f"coordination.hubs.{self.hub_id}.status",
            'shutdown',
            distributed=True
        )
        
        self.logger.info(f"Coordination Hub {self.hub_id} shutdown complete")


# Global coordination hub instance
_global_coordination_hub: Optional[RedisCoordinationHub] = None


async def get_coordination_hub(**kwargs) -> RedisCoordinationHub:
    """Get global coordination hub instance."""
    global _global_coordination_hub
    
    if _global_coordination_hub is None:
        _global_coordination_hub = RedisCoordinationHub(**kwargs)
        await _global_coordination_hub.initialize()
    
    return _global_coordination_hub


async def create_coordination_hub(**kwargs) -> RedisCoordinationHub:
    """Create new coordination hub instance."""
    hub = RedisCoordinationHub(**kwargs)
    await hub.initialize()
    return hub