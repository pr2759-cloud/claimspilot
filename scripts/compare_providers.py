"""Run the same 5 claims through each provider and print the results side by side."""
from api.triage import TriageError, triage_claim

# With one key only, compare two models of the same provider instead:
# put (provider, model) pairs here, e.g. a small and a large model.
RUNS = [("anthropic", None), ("openai", None)]  # None = the model from .env

CLAIMS = [
    "Hail last Tuesday dented my car's hood and cracked the windshield. It's still drivable.",
    "Grease fire in my kitchen tonight. The fire department came, the cabinets and microwave "
    "are destroyed, and we're staying with family because of the smoke smell.",
    "My neighbor's kid slipped on the ice on my front steps and broke his wrist. "
    "His parents are asking who pays the hospital bill.",
    "Someone stole my bike from my locked garage, I think over the weekend.",
    "Water damage.",
]

for i, claim in enumerate(CLAIMS, start=1):
    print(f"\nClaim {i}: {claim[:70]}...")
    for provider, model in RUNS:
        try:
            r = triage_claim(claim, provider=provider, model=model)
            print(f"  {provider:10} {r.category:16} {r.severity:9} missing={len(r.missing_info)}")
        except TriageError:
            print(f"  {provider:10} INVALID after retry")