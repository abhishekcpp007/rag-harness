from rag.answer import ABSTAIN, _enforce_citations, answer_question
from rag.llm import ExtractiveLLM, Generation, get_llm


class FakeLLM:
    name = "fake"

    def __init__(self, text):
        self._text = text

    def generate(self, question, blocks):
        return Generation(self._text, [], self.name)


def test_a_sentence_without_a_citation_is_dropped():
    kept, dropped = _enforce_citations("Grounded claim [1] Invented claim.", 2)
    assert kept == "Grounded claim [1]"
    assert dropped == 1


def test_a_citation_pointing_past_the_context_is_dropped():
    kept, dropped = _enforce_citations("Real [1] Hallucinated block [9]", 3)
    assert kept == "Real [1]"
    assert dropped == 1


def test_everything_uncited_becomes_an_abstain(retriever, indexed):
    a = answer_question("What time does the settlement sweep run in India?", retriever,
                        FakeLLM("The sweep runs at midnight. I am quite sure."), indexed)
    assert a.abstained
    assert a.text == ABSTAIN
    assert a.dropped_sentences >= 1


def test_the_model_can_say_it_does_not_know(retriever, indexed):
    a = answer_question("What time does the settlement sweep run in India?", retriever,
                        FakeLLM("NOT_IN_CORPUS"), indexed)
    assert a.abstained


def test_an_unanswerable_question_never_reaches_the_model(retriever, indexed):
    a = answer_question("How do I issue a virtual card to an employee?", retriever,
                        FakeLLM("Go to Cards and press New. [1]"), indexed)
    assert a.abstained
    assert a.backend == "abstain"


def test_a_real_answer_carries_citations_that_resolve(retriever, indexed):
    a = answer_question("What time does the settlement sweep run in India?", retriever,
                        get_llm("extractive"), indexed)
    assert not a.abstained
    assert a.citations
    assert all(c["chunk_id"] and c["doc_id"] for c in a.citations)
    assert "23:40" in a.text


def test_the_extractive_generator_only_repeats_the_corpus():
    gen = ExtractiveLLM().generate("what is the sweep time", ["Settlement :: The sweep runs at 23:40 IST."])
    assert "23:40 IST" in gen.text
    assert gen.text.endswith("[1]")
