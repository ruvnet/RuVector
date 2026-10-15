# Decision: validate all LangChain batch identities before inserting

The adapter rejected IDs already in the store but accepted repeated IDs within
one add_texts call. Two native rows could share one LangChain ID; deleting that
ID removed only the last row and left the earlier row without a forward mapping.

Seed a seen-ID set from existing identities and extend it while validating the
batch. Reject collisions before embedding or native mutation. Distinct explicit
IDs and generated IDs keep their existing behavior. The regression exercises
public add/search/delete calls backed by the published native extension and
checks native row counts. Existing-ID rejection also preserves the original row.
No persistent mapping format or LangChain upsert contract is introduced.
