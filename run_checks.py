 # Temporary verification script - click the > Run button (top-right), then delete.
from src.config import load_config
from src.generate.answer import answer_question

cfg = load_config()
questions = [
      "When should fluoroquinolone antibiotics now be prescribed?",     # expect grounded + cited
      "What rare neurological risk has been linked to pseudoephedrine?", # expect grounded + cited
      "What is the recommended ibuprofen dose for a tension headache?",  # expect a REFUSAL
      ]
for q in questions:
      r = answer_question(q, cfg)
      print("\n" + "=" * 70)
      print("Q:", q)
      print("-" * 70)
      print(r["answer"])
      print("\nSources returned:", r["sources"])