import json
from app.agent.agent import ClinicalTextClarifierAgent
from app.agent.schema import Agent1Request
from app.agent.normalizer import normalize_clinical_text
from app.agent.extractor import extract_deterministic_entities

text = "im 34 male, fever like 101/102, throat hurts, coughing a lot, started 2 days ago, no meds, no past illness"
print("Raw input:", text)

norm = normalize_clinical_text(text)
print("\nNormalized input:", norm)

det, understood = extract_deterministic_entities(norm)
print("\nDeterministic extraction:")
print("Understood:", understood)
print("Result:", det)

agent = ClinicalTextClarifierAgent()
resp = agent.process(Agent1Request(text=text))
print("\nAgent Full Process Response:")
print(json.dumps(resp.model_dump(), indent=2))
