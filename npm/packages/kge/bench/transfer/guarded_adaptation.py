"""Apply a trusted local candidate only after explicit validation acceptance."""
from validation_gate import evaluate_queries,load_adapted,select


def apply_candidate(adapter,checkpoint,expected_sha,queries,truths):
    """Sequential offline operation; restore incumbent on rejection or exception.

    Caller supplies validation-only queries and support/validation truths,
    preferably through official_validation. Never pass test rows here.
    The incumbent adapter should contain the frozen pretrained weights.
    """
    incumbent={k:v.detach().clone() for k,v in adapter.model.state_dict().items()}
    accepted=False
    try:
        adapter.model.eval()
        baseline=evaluate_queries(adapter,queries,truths)
        load_adapted(adapter,checkpoint,expected_sha)
        candidate=evaluate_queries(adapter,queries,truths)
        decision=select(baseline,candidate)
        accepted=decision["selected"]=="adapted"
        return dict(decision=decision,validationFrozen=baseline,validationAdapted=candidate)
    finally:
        if not accepted:
            adapter.model.load_state_dict(incumbent,strict=True)
        adapter.model.eval()
