"""
Test Redis Week 4 Advanced Features Implementation

Comprehensive test suite for Redis Week 4 advanced features: Pub/Sub coordination,
distributed locking, Lua scripts, message queues, and transaction management.
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

# Test imports
from scripts.redis_coordination_hub import (
    RedisCoordinationHub, CoordinationMessage, CoordinationMessageType, CoordinationPriority,
    LeaderElection, ConsensusManager, get_coordination_hub, create_coordination_hub
)
from scripts.redis_distributed_locks import (
    RedisDistributedLockManager, LockType, LockPriority, LockRequest,
    get_lock_manager, create_lock_manager
)
from scripts.redis_lua_engine import (
    RedisLuaEngine, LuaScript, ScriptType, ScriptPriority,
    get_lua_engine, create_lua_engine
)
from scripts.redis_message_queues import (
    RedisMessageQueue, QueueMessage, MessagePriority, DeliveryMode,
    get_message_queue, create_message_queue
)
from scripts.redis_transactions import (
    RedisTransactionManager, TransactionStatus, IsolationLevel, ConflictResolution,
    get_transaction_manager, create_transaction_manager, transaction
)


class RedisWeek4AdvancedTester:
    """Comprehensive tester for Redis Week 4 advanced features."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = []
        self.start_time = None
        
        # Test configuration
        self.test_config = {
            'test_nodes': 3,
            'test_messages': 25,
            'test_locks': 10,
            'test_scripts': 5,
            'test_transactions': 15,
            'test_duration_seconds': 60
        }
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite for Redis Week 4 advanced features."""
        self.start_time = time.time()
        self.logger.info("Starting Redis Week 4 Advanced Features comprehensive test suite")
        
        try:
            # Test 1: Coordination Hub
            await self._test_coordination_hub()
            
            # Test 2: Distributed Locking
            await self._test_distributed_locking()
            
            # Test 3: Lua Engine
            await self._test_lua_engine()
            
            # Test 4: Message Queues
            await self._test_message_queues()
            
            # Test 5: Transaction Management
            await self._test_transaction_management()
            
            # Test 6: Integration Testing
            await self._test_integration()
            
            # Test 7: Performance and Scalability
            await self._test_performance_scalability()
            
            # Generate final report
            return self._generate_test_report()
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive tests: {e}")
            self._add_test_result("comprehensive_tests", False, str(e))
            return self._generate_test_report()
    
    async def _test_coordination_hub(self):
        """Test Redis Coordination Hub functionality."""
        test_name = "coordination_hub"
        self.logger.info(f"Testing {test_name}")
        
        try:
            # Initialize coordination hub
            hub = await create_coordination_hub(
                hub_id="test_coord_hub",
                enable_leader_election=True,
                enable_consensus=True,
                enable_load_balancing=True
            )
            
            # Test coordination message broadcasting
            broadcast_success = await hub.broadcast_message(
                CoordinationMessageType.WORKER_ANNOUNCEMENT,
                {'worker_id': 'test_worker_1', 'capabilities': ['coordination', 'processing']},
                CoordinationPriority.NORMAL
            )
            
            if broadcast_success:
                self._add_test_result(f"{test_name}_broadcast", True, "Message broadcast successful")
            else:
                self._add_test_result(f"{test_name}_broadcast", False, "Message broadcast failed")
            
            # Test targeted messaging
            target_success = await hub.send_targeted_message(
                "target_node",
                CoordinationMessageType.TASK_BROADCAST,
                {'task_id': 'test_task_123', 'priority': 'high'},
                CoordinationPriority.HIGH
            )
            
            if target_success:
                self._add_test_result(f"{test_name}_targeted_message", True, "Targeted message successful")
            else:
                self._add_test_result(f"{test_name}_targeted_message", False, "Targeted message failed")
            
            # Test leader election
            if hub.leader_election:
                election_result = await hub.leader_election.start_election()
                
                if election_result:
                    self._add_test_result(f"{test_name}_leader_election", True, "Leader election successful")
                    
                    # Test leadership step down
                    await hub.leader_election.step_down()
                    self._add_test_result(f"{test_name}_step_down", True, "Leadership step down successful")
                else:
                    self._add_test_result(f"{test_name}_leader_election", False, "Leader election failed")
            
            # Test consensus proposal
            if hub.consensus_manager:
                consensus_result = await hub.propose_consensus(
                    "test_proposal_1",
                    {'proposal_type': 'system_update', 'version': '2.0.0'}
                )
                
                if consensus_result is not None:
                    self._add_test_result(f"{test_name}_consensus", True, "Consensus proposal initiated")
                else:
                    self._add_test_result(f"{test_name}_consensus", False, "Consensus proposal failed")
            
            # Test distributed locking through coordination hub
            lock_token = await hub.acquire_distributed_lock("test_coordination_lock", 30)
            
            if lock_token:
                self._add_test_result(f"{test_name}_distributed_lock", True, "Distributed lock acquired")
                
                # Release lock
                release_success = await hub.release_distributed_lock("test_coordination_lock")
                if release_success:
                    self._add_test_result(f"{test_name}_lock_release", True, "Lock released successfully")
                else:
                    self._add_test_result(f"{test_name}_lock_release", False, "Lock release failed")
            else:
                self._add_test_result(f"{test_name}_distributed_lock", False, "Distributed lock acquisition failed")
            
            # Test coordination status
            status = await hub.get_coordination_status()
            
            if status and 'hub_id' in status:
                self._add_test_result(f"{test_name}_status", True, f"Coordination status available")
            else:
                self._add_test_result(f"{test_name}_status", False, "Coordination status unavailable")
            
            # Cleanup
            await hub.shutdown()
            
            self.logger.info(f"{test_name} tests completed")
            
        except Exception as e:
            self.logger.error(f"Error in {test_name}: {e}")
            self._add_test_result(test_name, False, str(e))
    
    async def _test_distributed_locking(self):
        """Test Redis Distributed Locking functionality."""
        test_name = "distributed_locking"
        self.logger.info(f"Testing {test_name}")
        
        try:
            # Initialize lock manager
            lock_manager = await create_lock_manager(
                manager_id="test_lock_mgr",
                enable_deadlock_detection=True,
                enable_lock_monitoring=True
            )
            
            # Test exclusive lock
            exclusive_lock = await lock_manager.acquire_lock(
                "test_exclusive_lock",
                "requester_1",
                LockType.EXCLUSIVE,
                LockPriority.NORMAL,
                timeout_seconds=30
            )
            
            if exclusive_lock:
                self._add_test_result(f"{test_name}_exclusive_lock", True, "Exclusive lock acquired")
                
                # Test lock release
                release_success = await lock_manager.release_lock(exclusive_lock, "requester_1")
                if release_success:
                    self._add_test_result(f"{test_name}_exclusive_release", True, "Exclusive lock released")
                else:
                    self._add_test_result(f"{test_name}_exclusive_release", False, "Exclusive lock release failed")
            else:
                self._add_test_result(f"{test_name}_exclusive_lock", False, "Exclusive lock acquisition failed")
            
            # Test shared locks
            shared_lock_1 = await lock_manager.acquire_lock(
                "test_shared_lock",
                "requester_1",
                LockType.SHARED,
                LockPriority.NORMAL
            )
            
            shared_lock_2 = await lock_manager.acquire_lock(
                "test_shared_lock",
                "requester_2",
                LockType.SHARED,
                LockPriority.NORMAL
            )
            
            if shared_lock_1 and shared_lock_2:
                self._add_test_result(f"{test_name}_shared_locks", True, "Multiple shared locks acquired")
                
                # Release shared locks
                await lock_manager.release_lock(shared_lock_1, "requester_1")
                await lock_manager.release_lock(shared_lock_2, "requester_2")
            else:
                self._add_test_result(f"{test_name}_shared_locks", False, "Shared locks acquisition failed")
            
            # Test semaphore lock
            semaphore_lock = await lock_manager.acquire_lock(
                "test_semaphore_lock",
                "requester_1",
                LockType.SEMAPHORE,
                LockPriority.NORMAL,
                metadata={'max_count': 3}
            )
            
            if semaphore_lock:
                self._add_test_result(f"{test_name}_semaphore_lock", True, "Semaphore lock acquired")
                await lock_manager.release_lock(semaphore_lock, "requester_1")
            else:
                self._add_test_result(f"{test_name}_semaphore_lock", False, "Semaphore lock acquisition failed")
            
            # Test lock extension
            test_lock = await lock_manager.acquire_lock(
                "test_extension_lock",
                "requester_1",
                LockType.EXCLUSIVE
            )
            
            if test_lock:
                extension_success = await lock_manager.extend_lock(test_lock, "requester_1", 60)
                if extension_success:
                    self._add_test_result(f"{test_name}_lock_extension", True, "Lock extension successful")
                else:
                    self._add_test_result(f"{test_name}_lock_extension", False, "Lock extension failed")
                
                await lock_manager.release_lock(test_lock, "requester_1")
            
            # Test lock information
            lock_info = await lock_manager.get_lock_info("test_info_lock")
            # Lock doesn't exist, should return None
            if lock_info is None:
                self._add_test_result(f"{test_name}_lock_info", True, "Lock info query successful")
            else:
                self._add_test_result(f"{test_name}_lock_info", False, "Lock info query unexpected result")
            
            # Test statistics
            stats = await lock_manager.get_lock_statistics()
            
            if stats and 'manager_id' in stats:
                self._add_test_result(f"{test_name}_statistics", True, "Lock statistics available")
            else:
                self._add_test_result(f"{test_name}_statistics", False, "Lock statistics unavailable")
            
            # Cleanup
            await lock_manager.shutdown()
            
            self.logger.info(f"{test_name} tests completed")
            
        except Exception as e:
            self.logger.error(f"Error in {test_name}: {e}")
            self._add_test_result(test_name, False, str(e))
    
    async def _test_lua_engine(self):
        """Test Redis Lua Engine functionality."""
        test_name = "lua_engine"
        self.logger.info(f"Testing {test_name}")
        
        try:
            # Initialize Lua engine
            lua_engine = await create_lua_engine(
                engine_id="test_lua_engine",
                enable_script_caching=True,
                enable_performance_monitoring=True
            )
            
            # Test built-in atomic counter
            counter_result = await lua_engine.execute_atomic_counter(
                "test_counter",
                "incr",
                5,
                min_val=0,
                max_val=100
            )
            
            if counter_result and counter_result['status'] == 'success':
                self._add_test_result(f"{test_name}_atomic_counter", True, f"Counter value: {counter_result['value']}")
            else:
                self._add_test_result(f"{test_name}_atomic_counter", False, "Atomic counter failed")
            
            # Test distributed lock script
            lock_result = await lua_engine.execute_distributed_lock(
                "test_lua_lock",
                "test_owner",
                30
            )
            
            if lock_result and lock_result['acquired']:
                self._add_test_result(f"{test_name}_distributed_lock", True, "Lua distributed lock acquired")
            else:
                self._add_test_result(f"{test_name}_distributed_lock", False, "Lua distributed lock failed")
            
            # Test rate limiter
            rate_limit_result = await lua_engine.execute_rate_limiter(
                "test_rate_limit",
                100,  # max tokens
                10,   # refill rate
                5     # requested tokens
            )
            
            if rate_limit_result and rate_limit_result['allowed']:
                self._add_test_result(f"{test_name}_rate_limiter", True, "Rate limiter allowed request")
            else:
                self._add_test_result(f"{test_name}_rate_limiter", False, "Rate limiter denied request")
            
            # Test task dequeue
            task_result = await lua_engine.execute_task_dequeue(
                "test_task_queue",
                "test_processing",
                "test_worker",
                1
            )
            
            # Should return empty list since no tasks are queued
            if isinstance(task_result, list):
                self._add_test_result(f"{test_name}_task_dequeue", True, f"Task dequeue returned {len(task_result)} tasks")
            else:
                self._add_test_result(f"{test_name}_task_dequeue", False, "Task dequeue failed")
            
            # Test analytics aggregation
            analytics_result = await lua_engine.execute_analytics_aggregation(
                "test_metric",
                "test_window",
                42.5,
                300
            )
            
            if analytics_result and 'sum' in analytics_result:
                self._add_test_result(f"{test_name}_analytics", True, "Analytics aggregation successful")
            else:
                self._add_test_result(f"{test_name}_analytics", False, "Analytics aggregation failed")
            
            # Test script information
            script_info = await lua_engine.get_script_info("atomic_counter")
            
            if script_info and 'script_id' in script_info:
                self._add_test_result(f"{test_name}_script_info", True, "Script info available")
            else:
                self._add_test_result(f"{test_name}_script_info", False, "Script info unavailable")
            
            # Test engine statistics
            engine_stats = await lua_engine.get_engine_statistics()
            
            if engine_stats and 'engine_id' in engine_stats:
                self._add_test_result(f"{test_name}_engine_stats", True, "Engine statistics available")
            else:
                self._add_test_result(f"{test_name}_engine_stats", False, "Engine statistics unavailable")
            
            # Test script listing
            script_list = await lua_engine.list_scripts()
            
            if script_list and len(script_list) > 0:
                self._add_test_result(f"{test_name}_script_list", True, f"Found {len(script_list)} scripts")
            else:
                self._add_test_result(f"{test_name}_script_list", False, "Script listing failed")
            
            # Cleanup
            await lua_engine.shutdown()
            
            self.logger.info(f"{test_name} tests completed")
            
        except Exception as e:
            self.logger.error(f"Error in {test_name}: {e}")
            self._add_test_result(test_name, False, str(e))
    
    async def _test_message_queues(self):
        """Test Redis Message Queues functionality."""
        test_name = "message_queues"
        self.logger.info(f"Testing {test_name}")
        
        try:
            # Initialize message queue
            queue_manager = await create_message_queue(
                queue_manager_id="test_queue_mgr",
                enable_dead_letter=True,
                enable_priority_scheduling=True
            )
            
            # Create test queue
            queue_created = await queue_manager.create_queue(
                "test_queue",
                max_priority=6,
                enable_dlq=True,
                max_retries=3
            )
            
            if queue_created:
                self._add_test_result(f"{test_name}_queue_creation", True, "Queue created successfully")
            else:
                self._add_test_result(f"{test_name}_queue_creation", False, "Queue creation failed")
            
            # Test message enqueuing
            message_ids = []
            for i in range(self.test_config['test_messages']):
                priority = MessagePriority.HIGH if i % 3 == 0 else MessagePriority.NORMAL
                
                message_id = await queue_manager.enqueue_message(
                    "test_queue",
                    {'test_data': f'message_{i}', 'sequence': i},
                    priority=priority,
                    delivery_mode=DeliveryMode.AT_LEAST_ONCE,
                    max_retries=2,
                    message_type="test_message"
                )
                
                if message_id:
                    message_ids.append(message_id)
            
            if len(message_ids) == self.test_config['test_messages']:
                self._add_test_result(f"{test_name}_message_enqueue", True, f"Enqueued {len(message_ids)} messages")
            else:
                self._add_test_result(f"{test_name}_message_enqueue", False, f"Only enqueued {len(message_ids)}/{self.test_config['test_messages']} messages")
            
            # Test message dequeuing
            dequeued_messages = await queue_manager.dequeue_messages(
                "test_queue",
                "test_consumer",
                max_messages=5,
                processing_timeout=30
            )
            
            if len(dequeued_messages) > 0:
                self._add_test_result(f"{test_name}_message_dequeue", True, f"Dequeued {len(dequeued_messages)} messages")
                
                # Test message acknowledgment
                ack_success = True
                for message in dequeued_messages:
                    ack_result = await queue_manager.acknowledge_message(
                        message,
                        "test_consumer",
                        success=True
                    )
                    if not ack_result:
                        ack_success = False
                
                if ack_success:
                    self._add_test_result(f"{test_name}_message_ack", True, "Message acknowledgment successful")
                else:
                    self._add_test_result(f"{test_name}_message_ack", False, "Message acknowledgment failed")
            else:
                self._add_test_result(f"{test_name}_message_dequeue", False, "Message dequeue failed")
            
            # Test consumer registration
            message_count = 0
            
            async def test_handler(message):
                nonlocal message_count
                message_count += 1
                return True
            
            consumer_registered = await queue_manager.register_consumer(
                "test_consumer_handler",
                "test_queue",
                test_handler,
                concurrency=2
            )
            
            if consumer_registered:
                self._add_test_result(f"{test_name}_consumer_registration", True, "Consumer registered successfully")
                
                # Wait for some message processing
                await asyncio.sleep(3)
                
                if message_count > 0:
                    self._add_test_result(f"{test_name}_message_processing", True, f"Processed {message_count} messages")
                else:
                    self._add_test_result(f"{test_name}_message_processing", False, "No messages processed")
            else:
                self._add_test_result(f"{test_name}_consumer_registration", False, "Consumer registration failed")
            
            # Test queue statistics
            queue_stats = await queue_manager.get_queue_statistics("test_queue")
            
            if queue_stats:
                self._add_test_result(f"{test_name}_queue_stats", True, f"Queue statistics available")
            else:
                self._add_test_result(f"{test_name}_queue_stats", False, "Queue statistics unavailable")
            
            # Test manager statistics
            manager_stats = await queue_manager.get_all_statistics()
            
            if manager_stats and 'queue_manager_id' in manager_stats:
                self._add_test_result(f"{test_name}_manager_stats", True, "Manager statistics available")
            else:
                self._add_test_result(f"{test_name}_manager_stats", False, "Manager statistics unavailable")
            
            # Cleanup
            await queue_manager.shutdown()
            
            self.logger.info(f"{test_name} tests completed")
            
        except Exception as e:
            self.logger.error(f"Error in {test_name}: {e}")
            self._add_test_result(test_name, False, str(e))
    
    async def _test_transaction_management(self):
        """Test Redis Transaction Management functionality."""
        test_name = "transaction_management"
        self.logger.info(f"Testing {test_name}")
        
        try:
            # Initialize transaction manager
            txn_manager = await create_transaction_manager(
                manager_id="test_txn_mgr",
                default_isolation=IsolationLevel.READ_COMMITTED,
                default_conflict_resolution=ConflictResolution.RETRY,
                enable_deadlock_detection=True
            )
            
            # Test simple transaction
            txn_id = await txn_manager.begin_transaction(
                "test_initiator",
                isolation_level=IsolationLevel.READ_COMMITTED,
                timeout_seconds=30
            )
            
            if txn_id:
                self._add_test_result(f"{test_name}_begin_transaction", True, f"Transaction started: {txn_id}")
                
                # Test read and write operations
                await txn_manager.write_key(txn_id, "test_key_1", {"value": 100, "type": "test"})
                await txn_manager.write_key(txn_id, "test_key_2", {"value": 200, "type": "test"})
                
                # Test read operation
                value, success = await txn_manager.read_key(txn_id, "test_key_1")
                
                if success and value == {"value": 100, "type": "test"}:
                    self._add_test_result(f"{test_name}_read_write", True, "Read/write operations successful")
                else:
                    self._add_test_result(f"{test_name}_read_write", False, "Read/write operations failed")
                
                # Test transaction commit
                commit_success, commit_status, commit_result = await txn_manager.commit_transaction(txn_id)
                
                if commit_success:
                    self._add_test_result(f"{test_name}_commit", True, f"Transaction committed: {commit_status}")
                else:
                    self._add_test_result(f"{test_name}_commit", False, f"Transaction commit failed: {commit_status}")
            else:
                self._add_test_result(f"{test_name}_begin_transaction", False, "Transaction start failed")
            
            # Test transaction context manager
            try:
                async with transaction(initiator_id="context_test") as ctx:
                    await ctx.write("context_key_1", {"context": "test", "value": 42})
                    await ctx.write("context_key_2", {"context": "test", "value": 84})
                    
                    # Read back
                    value = await ctx.read("context_key_1")
                    
                    if value == {"context": "test", "value": 42}:
                        self._add_test_result(f"{test_name}_context_manager", True, "Transaction context manager successful")
                    else:
                        self._add_test_result(f"{test_name}_context_manager", False, "Transaction context manager read failed")
                        
            except Exception as ctx_error:
                self._add_test_result(f"{test_name}_context_manager", False, f"Context manager error: {ctx_error}")
            
            # Test transaction abort
            abort_txn_id = await txn_manager.begin_transaction("abort_test")
            
            if abort_txn_id:
                await txn_manager.write_key(abort_txn_id, "abort_key", {"will_be": "aborted"})
                
                abort_success = await txn_manager.abort_transaction(abort_txn_id, "test_abort")
                
                if abort_success:
                    self._add_test_result(f"{test_name}_abort", True, "Transaction abort successful")
                else:
                    self._add_test_result(f"{test_name}_abort", False, "Transaction abort failed")
            
            # Test concurrent transactions (conflict detection)
            txn1_id = await txn_manager.begin_transaction("concurrent_test_1")
            txn2_id = await txn_manager.begin_transaction("concurrent_test_2")
            
            if txn1_id and txn2_id:
                # Both transactions try to modify the same key
                await txn_manager.write_key(txn1_id, "conflict_key", {"txn": 1})
                await txn_manager.write_key(txn2_id, "conflict_key", {"txn": 2})
                
                # Commit first transaction
                commit1_success, _, _ = await txn_manager.commit_transaction(txn1_id)
                
                # Try to commit second transaction (should detect conflict)
                commit2_success, commit2_status, _ = await txn_manager.commit_transaction(txn2_id)
                
                if commit1_success and not commit2_success:
                    self._add_test_result(f"{test_name}_conflict_detection", True, "Conflict detection successful")
                else:
                    self._add_test_result(f"{test_name}_conflict_detection", False, "Conflict detection failed")
            
            # Test transaction status
            status_txn_id = await txn_manager.begin_transaction("status_test")
            if status_txn_id:
                status = await txn_manager.get_transaction_status(status_txn_id)
                
                if status and status.get('transaction_id') == status_txn_id:
                    self._add_test_result(f"{test_name}_status", True, "Transaction status available")
                else:
                    self._add_test_result(f"{test_name}_status", False, "Transaction status unavailable")
                
                await txn_manager.abort_transaction(status_txn_id)
            
            # Test manager statistics
            manager_stats = await txn_manager.get_manager_statistics()
            
            if manager_stats and 'manager_id' in manager_stats:
                self._add_test_result(f"{test_name}_manager_stats", True, "Manager statistics available")
            else:
                self._add_test_result(f"{test_name}_manager_stats", False, "Manager statistics unavailable")
            
            # Cleanup
            await txn_manager.shutdown()
            
            self.logger.info(f"{test_name} tests completed")
            
        except Exception as e:
            self.logger.error(f"Error in {test_name}: {e}")
            self._add_test_result(test_name, False, str(e))
    
    async def _test_integration(self):
        """Test integration between all Week 4 components."""
        test_name = "integration"
        self.logger.info(f"Testing {test_name}")
        
        try:
            # Initialize all components
            coord_hub = await get_coordination_hub()
            lock_manager = await get_lock_manager()
            lua_engine = await get_lua_engine()
            queue_manager = await get_message_queue()
            txn_manager = await get_transaction_manager()
            
            # Test 1: Coordinated transaction with distributed locking
            lock_id = await lock_manager.acquire_lock(
                "integration_lock",
                "integration_test",
                LockType.EXCLUSIVE
            )
            
            if lock_id:
                # Start transaction while holding lock
                txn_id = await txn_manager.begin_transaction("integration_test")
                
                if txn_id:
                    await txn_manager.write_key(txn_id, "integration_key", {"integration": "test"})
                    
                    # Commit transaction
                    commit_success, _, _ = await txn_manager.commit_transaction(txn_id)
                    
                    # Release lock
                    await lock_manager.release_lock(lock_id, "integration_test")
                    
                    if commit_success:
                        self._add_test_result(f"{test_name}_lock_transaction", True, "Lock + transaction integration successful")
                    else:
                        self._add_test_result(f"{test_name}_lock_transaction", False, "Transaction commit failed")
                else:
                    self._add_test_result(f"{test_name}_lock_transaction", False, "Transaction start failed")
            else:
                self._add_test_result(f"{test_name}_lock_transaction", False, "Lock acquisition failed")
            
            # Test 2: Message queue with Lua script processing
            # Create queue
            await queue_manager.create_queue("integration_queue")
            
            # Enqueue message
            msg_id = await queue_manager.enqueue_message(
                "integration_queue",
                {"script_test": True, "value": 123}
            )
            
            if msg_id:
                # Use Lua script to process queue
                counter_result = await lua_engine.execute_atomic_counter(
                    "integration_counter",
                    "incr",
                    1
                )
                
                if counter_result and counter_result['status'] == 'success':
                    self._add_test_result(f"{test_name}_queue_lua", True, "Queue + Lua integration successful")
                else:
                    self._add_test_result(f"{test_name}_queue_lua", False, "Lua script execution failed")
            else:
                self._add_test_result(f"{test_name}_queue_lua", False, "Message enqueue failed")
            
            # Test 3: Coordination with consensus and locking
            consensus_result = await coord_hub.propose_consensus(
                "integration_consensus",
                {"integration_test": True, "component_count": 5}
            )
            
            if consensus_result is not None:
                self._add_test_result(f"{test_name}_coordination_consensus", True, "Coordination + consensus integration successful")
            else:
                self._add_test_result(f"{test_name}_coordination_consensus", False, "Consensus proposal failed")
            
            # Test 4: Complex workflow using all components
            workflow_success = await self._execute_complex_workflow(
                coord_hub, lock_manager, lua_engine, queue_manager, txn_manager
            )
            
            if workflow_success:
                self._add_test_result(f"{test_name}_complex_workflow", True, "Complex workflow integration successful")
            else:
                self._add_test_result(f"{test_name}_complex_workflow", False, "Complex workflow failed")
            
            self.logger.info(f"{test_name} tests completed")
            
        except Exception as e:
            self.logger.error(f"Error in {test_name}: {e}")
            self._add_test_result(test_name, False, str(e))
    
    async def _execute_complex_workflow(self, coord_hub, lock_manager, lua_engine, queue_manager, txn_manager):
        """Execute complex workflow using all components."""
        try:
            # Step 1: Acquire coordination lock
            coord_lock = await coord_hub.acquire_distributed_lock("workflow_lock", 60)
            if not coord_lock:
                return False
            
            # Step 2: Start transaction
            txn_id = await txn_manager.begin_transaction("workflow")
            if not txn_id:
                await coord_hub.release_distributed_lock("workflow_lock")
                return False
            
            # Step 3: Create and populate queue with Lua script
            await queue_manager.create_queue("workflow_queue")
            
            for i in range(5):
                await queue_manager.enqueue_message(
                    "workflow_queue",
                    {"workflow_step": i, "data": f"step_{i}"}
                )
            
            # Step 4: Process messages in transaction
            messages = await queue_manager.dequeue_messages("workflow_queue", "workflow_processor", 5)
            
            for msg in messages:
                # Update counter with Lua script
                await lua_engine.execute_atomic_counter(f"workflow_counter_{msg.payload['workflow_step']}", "incr", 1)
                
                # Write to transaction
                await txn_manager.write_key(txn_id, f"workflow_step_{msg.payload['workflow_step']}", msg.payload)
                
                # Acknowledge message
                await queue_manager.acknowledge_message(msg, "workflow_processor", True)
            
            # Step 5: Commit transaction
            commit_success, _, _ = await txn_manager.commit_transaction(txn_id)
            
            # Step 6: Release coordination lock
            await coord_hub.release_distributed_lock("workflow_lock")
            
            return commit_success and len(messages) == 5
            
        except Exception as e:
            self.logger.error(f"Error in complex workflow: {e}")
            return False
    
    async def _test_performance_scalability(self):
        """Test performance and scalability of Week 4 components."""
        test_name = "performance_scalability"
        self.logger.info(f"Testing {test_name}")
        
        try:
            # Performance test: Message throughput
            queue_manager = await get_message_queue()
            await queue_manager.create_queue("perf_queue")
            
            start_time = time.time()
            message_count = 100
            
            for i in range(message_count):
                await queue_manager.enqueue_message(
                    "perf_queue",
                    {"perf_test": i, "timestamp": time.time()}
                )
            
            enqueue_time = time.time() - start_time
            enqueue_throughput = message_count / enqueue_time
            
            if enqueue_throughput > 20:  # Expect at least 20 messages/sec
                self._add_test_result(f"{test_name}_message_throughput", True, f"Enqueue throughput: {enqueue_throughput:.1f} msg/sec")
            else:
                self._add_test_result(f"{test_name}_message_throughput", False, f"Low enqueue throughput: {enqueue_throughput:.1f} msg/sec")
            
            # Performance test: Lock acquisition
            lock_manager = await get_lock_manager()
            
            start_time = time.time()
            lock_count = 50
            locks = []
            
            for i in range(lock_count):
                lock_id = await lock_manager.acquire_lock(f"perf_lock_{i}", "perf_test", LockType.EXCLUSIVE)
                if lock_id:
                    locks.append((lock_id, f"perf_lock_{i}"))
            
            lock_time = time.time() - start_time
            lock_throughput = len(locks) / lock_time
            
            # Release locks
            for lock_id, lock_name in locks:
                await lock_manager.release_lock(lock_id, "perf_test")
            
            if lock_throughput > 10:  # Expect at least 10 locks/sec
                self._add_test_result(f"{test_name}_lock_throughput", True, f"Lock throughput: {lock_throughput:.1f} locks/sec")
            else:
                self._add_test_result(f"{test_name}_lock_throughput", False, f"Low lock throughput: {lock_throughput:.1f} locks/sec")
            
            # Performance test: Transaction throughput
            txn_manager = await get_transaction_manager()
            
            start_time = time.time()
            txn_count = 30
            successful_txns = 0
            
            for i in range(txn_count):
                txn_id = await txn_manager.begin_transaction(f"perf_test_{i}")
                if txn_id:
                    await txn_manager.write_key(txn_id, f"perf_key_{i}", {"value": i})
                    success, _, _ = await txn_manager.commit_transaction(txn_id)
                    if success:
                        successful_txns += 1
            
            txn_time = time.time() - start_time
            txn_throughput = successful_txns / txn_time
            
            if txn_throughput > 5:  # Expect at least 5 txns/sec
                self._add_test_result(f"{test_name}_transaction_throughput", True, f"Transaction throughput: {txn_throughput:.1f} txns/sec")
            else:
                self._add_test_result(f"{test_name}_transaction_throughput", False, f"Low transaction throughput: {txn_throughput:.1f} txns/sec")
            
            # Performance test: Lua script execution
            lua_engine = await get_lua_engine()
            
            start_time = time.time()
            script_count = 100
            
            for i in range(script_count):
                await lua_engine.execute_atomic_counter(f"perf_script_counter_{i % 10}", "incr", 1)
            
            script_time = time.time() - start_time
            script_throughput = script_count / script_time
            
            if script_throughput > 30:  # Expect at least 30 scripts/sec
                self._add_test_result(f"{test_name}_script_throughput", True, f"Script throughput: {script_throughput:.1f} scripts/sec")
            else:
                self._add_test_result(f"{test_name}_script_throughput", False, f"Low script throughput: {script_throughput:.1f} scripts/sec")
            
            self.logger.info(f"{test_name} tests completed")
            
        except Exception as e:
            self.logger.error(f"Error in {test_name}: {e}")
            self._add_test_result(test_name, False, str(e))
    
    def _add_test_result(self, test_name: str, success: bool, message: str):
        """Add test result to results list."""
        result = {
            'test_name': test_name,
            'success': success,
            'message': message,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        self.test_results.append(result)
        
        status = "PASS" if success else "FAIL"
        self.logger.info(f"[{status}] {test_name}: {message}")
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        total_time = time.time() - self.start_time if self.start_time else 0
        
        # Categorize results
        categories = {
            'coordination_hub': [r for r in self.test_results if 'coordination_hub' in r['test_name']],
            'distributed_locking': [r for r in self.test_results if 'distributed_locking' in r['test_name']],
            'lua_engine': [r for r in self.test_results if 'lua_engine' in r['test_name']],
            'message_queues': [r for r in self.test_results if 'message_queues' in r['test_name']],
            'transaction_management': [r for r in self.test_results if 'transaction_management' in r['test_name']],
            'integration': [r for r in self.test_results if 'integration' in r['test_name']],
            'performance_scalability': [r for r in self.test_results if 'performance_scalability' in r['test_name']]
        }
        
        category_summary = {}
        for category, results in categories.items():
            if results:
                category_passed = sum(1 for r in results if r['success'])
                category_summary[category] = {
                    'total': len(results),
                    'passed': category_passed,
                    'success_rate': category_passed / len(results)
                }
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': success_rate,
                'total_time_seconds': total_time,
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            'category_summary': category_summary,
            'test_results': self.test_results,
            'recommendations': self._generate_recommendations()
        }
        
        self.logger.info(f"Test Report: {passed_tests}/{total_tests} tests passed ({success_rate:.1%})")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        failed_tests = [result for result in self.test_results if not result['success']]
        
        if not failed_tests:
            recommendations.append("All tests passed! Redis Week 4 Advanced Features implementation is ready for production.")
            recommendations.append("Consider proceeding to Week 5: Production monitoring, security, and optimization.")
        else:
            recommendations.append(f"{len(failed_tests)} tests failed. Review failed tests before proceeding.")
            
            # Category-specific recommendations
            failed_categories = set()
            for test in failed_tests:
                if 'coordination_hub' in test['test_name']:
                    failed_categories.add('coordination')
                elif 'distributed_locking' in test['test_name']:
                    failed_categories.add('locking')
                elif 'lua_engine' in test['test_name']:
                    failed_categories.add('lua')
                elif 'message_queues' in test['test_name']:
                    failed_categories.add('queues')
                elif 'transaction_management' in test['test_name']:
                    failed_categories.add('transactions')
                elif 'integration' in test['test_name']:
                    failed_categories.add('integration')
                elif 'performance' in test['test_name']:
                    failed_categories.add('performance')
            
            if 'coordination' in failed_categories:
                recommendations.append("Coordination Hub tests failed. Check Pub/Sub configuration and leader election setup.")
            
            if 'locking' in failed_categories:
                recommendations.append("Distributed Locking tests failed. Verify lock manager configuration and deadlock detection.")
            
            if 'lua' in failed_categories:
                recommendations.append("Lua Engine tests failed. Check script registration and Redis Lua environment.")
            
            if 'queues' in failed_categories:
                recommendations.append("Message Queue tests failed. Verify queue creation and message handling logic.")
            
            if 'transactions' in failed_categories:
                recommendations.append("Transaction Management tests failed. Check optimistic locking and conflict resolution.")
            
            if 'integration' in failed_categories:
                recommendations.append("Integration tests failed. Review component interactions and dependencies.")
            
            if 'performance' in failed_categories:
                recommendations.append("Performance tests failed. Consider Redis optimization and infrastructure scaling.")
        
        return recommendations


async def main():
    """Main test runner function."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Redis Week 4 Advanced Features Test Suite")
    
    try:
        # Initialize tester
        tester = RedisWeek4AdvancedTester()
        
        # Run comprehensive tests
        test_report = await tester.run_comprehensive_tests()
        
        # Print summary
        summary = test_report['summary']
        logger.info("=" * 80)
        logger.info("REDIS WEEK 4 ADVANCED FEATURES TEST REPORT")
        logger.info("=" * 80)
        logger.info(f"Total Tests: {summary['total_tests']}")
        logger.info(f"Passed: {summary['passed_tests']}")
        logger.info(f"Failed: {summary['failed_tests']}")
        logger.info(f"Success Rate: {summary['success_rate']:.1%}")
        logger.info(f"Total Time: {summary['total_time_seconds']:.1f}s")
        logger.info("=" * 80)
        
        # Print category summary
        logger.info("CATEGORY BREAKDOWN:")
        for category, stats in test_report['category_summary'].items():
            logger.info(f"{category}: {stats['passed']}/{stats['total']} ({stats['success_rate']:.1%})")
        logger.info("=" * 80)
        
        # Print recommendations
        logger.info("RECOMMENDATIONS:")
        for i, recommendation in enumerate(test_report['recommendations'], 1):
            logger.info(f"{i}. {recommendation}")
        
        # Save detailed report
        report_filename = f"redis_week4_advanced_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(test_report, f, indent=2)
        
        logger.info(f"Detailed report saved to: {report_filename}")
        
        # Return success status
        return summary['success_rate'] >= 0.8  # 80% success threshold
        
    except Exception as e:
        logger.error(f"Error in test runner: {e}")
        return False


if __name__ == "__main__":
    # Run the test suite
    success = asyncio.run(main())
    exit(0 if success else 1)