"""Expert guidance system for curated Ethereum interpretations.

This module provides curated expert knowledge that goes beyond
what's explicitly stated in the specs. It captures nuances,
gotchas, and insights from EF discussions.

NOTE: This is a stub that can be populated incrementally.
The full guidance document (~50 pages) is not currently available.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GuidanceEntry:
    """A curated guidance entry."""

    topic: str
    summary: str
    key_points: list[str]
    gotchas: list[str]
    references: list[str]


# Guidance database - populated incrementally
# Each entry captures nuanced understanding beyond the raw specs
GUIDANCE_DB: dict[str, GuidanceEntry] = {
    "churn": GuidanceEntry(
        topic="Validator Churn and Queue Mechanics",
        summary="Electra uses Gwei-based churn with SEPARATE pools for activations and exits.",
        key_points=[
            "Electra switched from counting validators to counting Gwei for churn",
            "MIN_PER_EPOCH_CHURN_LIMIT_ELECTRA = 128 ETH (floor for each pool)",
            "MAX_PER_EPOCH_ACTIVATION_EXIT_CHURN_LIMIT = 256 ETH",
            "ACTIVATIONS and EXITS have SEPARATE 256 ETH/epoch caps (not shared!)",
            "Total theoretical throughput: 512 ETH/epoch (256 activation + 256 exit)",
            "Partial withdrawals, full exits (CL and EL-triggered) share the EXIT pool",
            "Consolidations use remaining churn after activation/exit allocation",
        ],
        gotchas=[
            "Spec naming misleading - 'get_activation_exit_churn_limit' sounds shared but separate",
            "Partial withdrawals DO consume from exit churn - they share queue with full exits",
            "The difference is outcome: partial withdrawals skip sweep, validator stays active",
            "A validator with pending partial withdrawals cannot full exit until processed",
            "EL-triggered exits (EIP-7002) and CL voluntary exits use identical queue mechanics",
            "With ~34M ETH staked, both pools capped at 256 ETH (dynamic calc would be ~519 ETH)",
        ],
        references=[
            "specs/electra/beacon-chain.md - process_withdrawal_request",
            "specs/electra/beacon-chain.md - compute_exit_epoch_and_update_churn",
            "specs/electra/beacon-chain.md - get_activation_exit_churn_limit",
            "EIP-7002 - Execution layer triggerable exits",
        ],
    ),
    "slashing": GuidanceEntry(
        topic="Slashing Mechanics",
        summary="Slashing penalties evolved across forks, with Electra adjusting for higher MaxEB.",
        key_points=[
            "Two slashable offenses: proposer double-sign, attester double-vote/surround-vote",
            "Immediate penalty: effective_balance // MIN_SLASHING_PENALTY_QUOTIENT",
            "Proportional penalty: processed later based on total slashed in ~36-day window",
            "Phase0: quotient=128, multiplier=1 (conservative)",
            "Altair: quotient=64, multiplier=2",
            "Bellatrix: quotient=32, multiplier=3 (final security values)",
            "Electra: quotient=4096 (adjusted for MaxEB up to 2048 ETH)",
        ],
        gotchas=[
            "Electra's 1/4096 quotient looks larger but it's a SMALLER initial penalty",
            "This incentivizes consolidation to larger validators (less penalty per ETH)",
            "A 2048 ETH validator at 1/32 would lose 64 ETH immediately - too harsh",
            "At 1/4096, a 2048 ETH validator loses only 0.5 ETH initially",
            "Proportional penalty still punishes correlated slashing events heavily",
            "Withdrawable epoch extended by EPOCHS_PER_SLASHINGS_VECTOR (~36 days)",
        ],
        references=[
            "specs/phase0/beacon-chain.md - slash_validator",
            "specs/electra/beacon-chain.md - MIN_SLASHING_PENALTY_QUOTIENT_ELECTRA",
        ],
    ),
    "maxeb": GuidanceEntry(
        topic="Maximum Effective Balance (MaxEB)",
        summary="Electra increased MaxEB from 32 ETH to 2048 ETH for compounding validators.",
        key_points=[
            "MIN_ACTIVATION_BALANCE = 32 ETH (unchanged)",
            "MAX_EFFECTIVE_BALANCE_ELECTRA = 2048 ETH (for 0x02 credentials)",
            "Validators can consolidate multiple 32 ETH validators into one",
            "Reduces total validator count, improves network efficiency",
            "Compounding credentials (0x02) required for balances > 32 ETH",
            "0x01 credentials remain capped at 32 ETH effective balance",
        ],
        gotchas=[
            "Consolidation consumes from consolidation churn (separate from exit churn)",
            "Source validator exits, target validator receives the balance",
            "Both validators must have same withdrawal credentials",
            "Slashed validators cannot complete pending consolidations",
        ],
        references=[
            "specs/electra/beacon-chain.md - MAX_EFFECTIVE_BALANCE_ELECTRA",
            "specs/electra/beacon-chain.md - process_pending_consolidations",
            "EIP-7251 - Increase MAX_EFFECTIVE_BALANCE",
        ],
    ),
    "withdrawals": GuidanceEntry(
        topic="Withdrawal Mechanics",
        summary="Capella enabled withdrawals; Electra added partial withdrawal requests via EL.",
        key_points=[
            "Automatic withdrawals: sweep processes validators with excess balance",
            "Full withdrawals: exited validators get full balance returned",
            "Partial withdrawals (Electra): request specific amount via EL",
            "EIP-7002: Execution layer can trigger both partial withdrawals and full exits",
            "Withdrawal amount = 0 signals full exit request",
            "Withdrawal amount > 0 signals partial withdrawal request",
        ],
        gotchas=[
            "Partial withdrawal requests consume exit queue churn",
            "They don't go through the normal withdrawal sweep",
            "pending_partial_withdrawals queue is separate from exit queue",
            "But the CHURN is shared - that's the key nuance",
            "MAX_PENDING_PARTIALS_PER_WITHDRAWALS_SWEEP = 8 per payload",
        ],
        references=[
            "specs/capella/beacon-chain.md - process_withdrawals",
            "specs/electra/beacon-chain.md - process_withdrawal_request",
            "EIP-7002 - Execution layer triggerable exits",
        ],
    ),
    # ===================
    # MEV / PBS
    # ===================
    "mev": GuidanceEntry(
        topic="MEV (Maximal Extractable Value)",
        summary="Value extracted by reordering, inserting, or censoring transactions in a block.",
        key_points=[
            "MEV sources: DEX arbitrage, liquidations, sandwich attacks, NFT sniping",
            "Pre-merge: miners extracted MEV directly",
            "Post-merge: validators propose blocks, can extract MEV or outsource to builders",
            "MEV-boost: middleware connecting validators to block builders",
            "Flashbots: primary MEV infrastructure provider (relay, builder, protect RPC)",
            "Current MEV flow: Searchers → Builders → Relays → Validators",
            "~90% of Ethereum blocks use MEV-boost as of 2025",
        ],
        gotchas=[
            "MEV is not inherently bad - arbitrage improves market efficiency",
            "Harmful MEV: sandwich attacks extract value from users",
            "Flashbots Protect: RPC that shields users from sandwich attacks",
            "MEV-Share: protocol for sharing MEV profits with users",
            "Relays are trusted - they could theoretically censor or steal",
            "Builder dominance is a centralization concern",
        ],
        references=[
            "flashbots/mev-boost - MEV middleware",
            "flashbots/builder - Block builder reference",
            "ethereum/builder-specs - Builder API",
            "https://writings.flashbots.net/",
        ],
    ),
    "pbs": GuidanceEntry(
        topic="PBS (Proposer-Builder Separation)",
        summary="Protocol separating proposing (validators) from building (specialized actors).",
        key_points=[
            "Current state: out-of-protocol PBS via MEV-boost",
            "Proposer: validator selected to propose a block, chooses highest bid",
            "Builder: specialized actor that constructs blocks, bids for inclusion",
            "Relay: trusted intermediary between builders and proposers",
            "Builder commits to block header, proposer signs without seeing contents",
            "If builder doesn't deliver, proposer can propose empty block (no penalty)",
            "ePBS (enshrined PBS): PBS built into the protocol itself",
        ],
        gotchas=[
            "Current PBS relies on trusted relays - protocol doesn't enforce builder honesty",
            "Relay operators see full block contents - potential for frontrunning",
            "Builder concentration: top 3 builders often build >90% of blocks",
            "ePBS aims to make PBS trustless via protocol-level commitments",
            "Inclusion lists (IL): mechanism for proposers to guarantee certain tx inclusion",
            "ePBS is complex - still being researched and specified",
        ],
        references=[
            "ethereum/builder-specs - Current builder API",
            "specs/_features/eip7732 - ePBS specification",
            "EIP-7732 - Enshrined Proposer-Builder Separation",
            "EIP-7547 - Inclusion lists",
        ],
    ),
    "epbs": GuidanceEntry(
        topic="ePBS (Enshrined Proposer-Builder Separation)",
        summary="Protocol-native PBS removing reliance on trusted relays.",
        key_points=[
            "EIP-7732: Enshrined Proposer-Builder Separation",
            "Removes need for trusted relays",
            "Two-slot mechanism: slot N commits to builder, slot N+1 reveals block",
            "Execution attestations: attest to execution payload validity",
            "Unconditional payment: builder pays proposer even if block invalid",
            "PTC (Payload Timeliness Committee): attests to payload delivery",
            "Expected: post-Fulu fork (2026+)",
        ],
        gotchas=[
            "Increases slot time considerations (two-phase commit)",
            "Complexity: new committee type, new attestation type",
            "Still doesn't solve builder centralization directly",
            "Inclusion lists (IL) complement ePBS for censorship resistance",
            "Execution tickets: alternative design being researched",
            "MEV burn: proposal to burn MEV instead of paying proposers",
        ],
        references=[
            "EIP-7732 - ePBS specification",
            "specs/_features/eip7732 - ePBS beacon chain changes",
            "EIP-7547 - Inclusion lists (related)",
            "https://ethresear.ch/t/unbundling-pbs/",
        ],
    ),
    "mev_boost": GuidanceEntry(
        topic="MEV-Boost Architecture",
        summary="Out-of-protocol PBS implementation connecting validators to block builders.",
        key_points=[
            "Sidecar to consensus client (runs alongside beacon node)",
            "Validator registers with relays at epoch boundaries",
            "At block proposal time, MEV-boost queries registered relays",
            "Relays return bids from builders (block header + bid amount)",
            "MEV-boost selects highest bid, returns to validator",
            "Validator signs blinded block (header only)",
            "Relay reveals full block to network after signature",
        ],
        gotchas=[
            "Validator never sees block contents until after signing",
            "Trust assumption: relay will deliver valid block",
            "If relay fails, validator can fall back to local block building",
            "Multiple relays can be configured for redundancy",
            "Relay selection affects block censorship (some relays OFAC-compliant)",
            "~6 major relays: Flashbots, bloXroute, Ultrasound, Agnostic, Aestus, Titan",
            "min-bid flag: minimum bid to accept from builders (default: 0)",
        ],
        references=[
            "flashbots/mev-boost - Implementation",
            "flashbots/mev-boost-relay - Relay implementation",
            "https://boost.flashbots.net/ - Flashbots relay",
            "https://mevboost.pics/ - MEV-boost statistics",
        ],
    ),
    "flashbots": GuidanceEntry(
        topic="Flashbots Ecosystem",
        summary="Research and development org building MEV infrastructure.",
        key_points=[
            "Founded 2020 to mitigate MEV externalities",
            "MEV-Geth (deprecated): early solution for miners",
            "MEV-Boost: current standard for validators (post-merge)",
            "Flashbots Relay: largest MEV relay by volume",
            "Flashbots Builder: reference block builder implementation",
            "Flashbots Protect: RPC endpoint protecting users from frontrunning",
            "MEV-Share: protocol for MEV redistribution to users",
            "SUAVE: upcoming chain for decentralized block building",
        ],
        gotchas=[
            "Flashbots relay was OFAC-compliant (censored Tornado Cash txs) until 2023",
            "Switched to 'neutral' mode but still debated in community",
            "rbuilder: new Rust builder for better performance",
            "MEV-Share can reduce harmful MEV but adds complexity",
            "SUAVE is a separate chain (not Ethereum L1/L2)",
            "Flashbots is a for-profit company despite open source contributions",
        ],
        references=[
            "flashbots/mev-boost",
            "flashbots/builder",
            "flashbots/mev-boost-relay",
            "flashbots/rbuilder",
            "https://docs.flashbots.net/",
        ],
    ),
}


def get_expert_guidance(topic: str) -> GuidanceEntry | None:
    """
    Get expert guidance for a topic.

    Searches for exact match first, then partial matches.
    """
    topic_lower = topic.lower()

    # Exact match
    if topic_lower in GUIDANCE_DB:
        return GUIDANCE_DB[topic_lower]

    # Partial match
    for key, entry in GUIDANCE_DB.items():
        if topic_lower in key or key in topic_lower:
            return entry
        if topic_lower in entry.summary.lower():
            return entry

    return None


def list_guidance_topics() -> list[str]:
    """List all available guidance topics."""
    return list(GUIDANCE_DB.keys())


def add_guidance(entry: GuidanceEntry) -> None:
    """Add a new guidance entry."""
    GUIDANCE_DB[entry.topic.lower()] = entry
