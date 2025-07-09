# 11-Phase Research System Flow Diagram

## Visual Flow Representation

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 0: MARKET TIMING                       │
│  • Check economic indicators, VC climate, emerging tech         │
│  • Score: Timing (1-10) + Trends (1-10) / 2                   │
│  • Abort if combined < 6.5 (balanced) or < 7.5 (strict)       │
└────────────────────────────┬───────────────────────────────────┘
                             │ If favorable
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 1: PROBLEM DISCOVERY                    │
│  • AI generates unique research direction                       │
│  • Creates 10+ search queries                                  │
│  • Discovers 8-10 problems with severity ≥ 7/10               │
│  • Generates 5-7 features per problem                         │
│  • Deduplicates similar problems                              │
└────────────────────────────┬───────────────────────────────────┘
                             │ 8-10 problems
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PHASE 2: MARKET VALIDATION                     │
│  • Batch process (10 at a time)                               │
│  • Search for existing solutions & limitations                 │
│  • Validate willingness to pay                                │
│  • Min score: 6.0 (balanced) or 7.0 (strict)                 │
│  • KILL: "Insufficient market validation"                      │
└────────────────────────────┬───────────────────────────────────┘
                             │ ~6-8 survive
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 PHASE 2.5: USER RESEARCH                        │
│  • Find 10+ real user stories                                 │
│  • Measure pain severity (1-10)                               │
│  • Validate emotional & financial impact                       │
│  • Min score: 6.5 (balanced) or 7.5 (strict)                 │
│  • KILL: "Insufficient user pain"                             │
└────────────────────────────┬───────────────────────────────────┘
                             │ ~4-6 survive
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               PHASE 2.75: REVENUE VALIDATION                    │
│  • Research similar product pricing                            │
│  • Analyze competitor revenue models                           │
│  • Estimate ARPU potential                                    │
│  • Min score: 7.0 (balanced) or 8.0 (strict)                 │
│  • KILL: "No proven revenue potential"                         │
└────────────────────────────┬───────────────────────────────────┘
                             │ ~3-4 survive
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               PHASE 3: COMPETITION ANALYSIS                     │
│  • Identify direct & indirect competitors                      │
│  • Assess market saturation                                    │
│  • Find differentiation opportunities                          │
│  • Score: 1-3 (low) = good, 7-10 (high) = bad                │
│  • Min score: 6.0 (balanced) or 7.0 (strict)                 │
└────────────────────────────┬───────────────────────────────────┘
                             │ ~2-3 survive
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 3.5: TECHNICAL FEASIBILITY                   │
│  • Detailed architecture analysis                              │
│  • Estimate development time & costs                          │
│  • Validate 90%+ automation possible                          │
│  • Min score: 7.0 (balanced) or 8.0 (strict)                 │
│  • Target: MVP in <4 weeks                                    │
└────────────────────────────┬───────────────────────────────────┘
                             │ ~2-3 survive
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PHASE 4: MARKET SIZING                         │
│  • Calculate TAM, SAM, SOM                                     │
│  • Create path to $10K MRR                                    │
│  • Validate customer acquisition strategy                      │
│  • Min score: 7.0 (balanced) or 8.0 (strict)                 │
│  • Target: 1000+ potential customers                          │
└────────────────────────────┬───────────────────────────────────┘
                             │ ~1-2 survive
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 PHASE 4.5: UNIT ECONOMICS                       │
│  • Calculate CAC and LTV                                       │
│  • Validate gross margins (>70%)                              │
│  • Ensure LTV:CAC > 3:1                                       │
│  • Min score: 7.5 (balanced) or 8.5 (strict)                 │
│  • Payback period <12 months                                  │
└────────────────────────────┬───────────────────────────────────┘
                             │ ~1-2 survive
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                PHASE 4.75: MVP SIMULATION                       │
│  • Define core MVP features                                    │
│  • Simulate development timeline                               │
│  • Project early adoption curve                                │
│  • Min score: 7.5 (balanced) or 8.5 (strict)                 │
│  • Revenue projection for months 1-6                           │
└────────────────────────────┬───────────────────────────────────┘
                             │ 1 survives
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 PHASE 5: FINAL SELECTION                        │
│  • Compare all surviving opportunities                         │
│  • Calculate success probability:                              │
│    - User pain (15%)                                          │
│    - Revenue validation (15%)                                  │
│    - Market size (15%)                                        │
│    - Competition (10%)                                        │
│    - Technical feasibility (15%)                              │
│    - Unit economics (15%)                                     │
│    - Market timing (10%)                                      │
│    - Team capability (5%)                                     │
│  • Select top opportunity                                      │
└────────────────────────────┬───────────────────────────────────┘
                             │ Best opportunity
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               PHASE 6: PRE-BUILD VALIDATION                     │
│  • Final sanity check                                          │
│  • Review all analysis                                         │
│  • Check for market changes                                    │
│  • GO/NO-GO decision                                          │
│  • Generate project name                                       │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
                    🎯 FINAL OPPORTUNITY
```

## Data Structure Evolution Through Phases

### Phase 1 Output Structure
```json
{
  "id": "opp_0",
  "problem_description": "Beginners struggle with debugging...",
  "affected_audience": "Beginner coders",
  "current_pain": "Hours wasted on simple errors",
  "severity_score": 8.5,
  "transaction_potential": "Per-error fix or subscription",
  "key_features": [
    "AI Error Explanation",
    "Code Fix Suggestions",
    "Learning Dashboard"
  ]
}
```

### Progressive Enhancement
Each phase adds new data:
- Phase 2: `validation_score`, `evidence`, `revenue_model`
- Phase 2.5: `user_stories`, `pain_severity`, `willingness_to_switch`
- Phase 2.75: `revenue_evidence`, `arpu_estimate`, `competitor_pricing`
- Phase 3: `competition_score`, `main_competitors`, `differentiation`
- Phase 3.5: `feasibility_score`, `mvp_timeline`, `tech_stack`, `costs`
- Phase 4: `market_score`, `tam`, `sam`, `som`, `path_to_10k`
- Phase 4.5: `unit_score`, `cac`, `ltv`, `gross_margin`
- Phase 4.75: `mvp_features`, `launch_strategy`, `revenue_projection`
- Phase 5: `success_probability`, `rank`, `risk_factors`
- Phase 6: `validation_status`, `project_name`, `go_no_go`

## Kill Points and Reasons

### Early Termination Triggers
1. **Phase 2**: Score < 6.0 → "Insufficient market validation"
2. **Phase 2.5**: Score < 6.5 → "Insufficient user pain"
3. **Phase 2.75**: Score < 7.0 → "No proven revenue potential"
4. **Phase 3**: Competition > 7 → "Market too competitive"
5. **Phase 3.5**: Score < 7.0 → "Not technically feasible"
6. **Phase 4**: Score < 7.0 → "Market too small"
7. **Phase 4.5**: Score < 7.5 → "Poor unit economics"
8. **Phase 4.75**: Score < 7.5 → "MVP not viable"

## Optimization Strategies

### 1. Caching Strategy
```
Query: "online learning frustrations 2024"
Cache Key: MD5("online learning frustrations YEAR")
Hit Rate Target: >30%
```

### 2. Batch Processing
```
Phase 1: Process 20 problems per API call
Phase 2: Process 10 validations per call
Phase 3+: Process 3-5 per call
```

### 3. Progressive Filtering
```
Start: 50 potential problems
After Phase 2: 30 opportunities (40% killed)
After Phase 3: 10 opportunities (67% killed)
After Phase 4: 5 opportunities (50% killed)
Final: 1 opportunity (80% killed)
Total Kill Rate: ~98%
```

## Success Metrics

### Efficiency Metrics
- **API Calls per Opportunity**: 10-20
- **Time per Complete Run**: 5-10 minutes
- **Cache Hit Rate**: 30-50%
- **Progressive Kill Rate**: 95-98%

### Quality Metrics
- **Success Probability**: >70% for selected opportunities
- **Feature Generation**: 5-7 concrete features per opportunity
- **Evidence Depth**: 10+ validation points per phase
- **Market Size Validation**: Path to $10K MRR required

## Configuration Tuning

### For More Opportunities
- Lower severity threshold (7→6)
- Reduce min scores by 0.5
- Increase phase limits
- Relax kill criteria

### For Higher Quality
- Raise severity threshold (7→8)
- Increase min scores by 0.5
- Reduce phase limits
- Stricter kill criteria

### Balanced Approach (Recommended)
- Severity: 7/10
- Progressive scores: 6.0-7.5
- Moderate phase limits
- 70% success probability target