from langchain_core.output_parsers import (CommaSeparatedListOutputParser,
                                           JsonOutputParser)


def test_CommaSeparatedListOutputParser():
    parser = CommaSeparatedListOutputParser()
    assert parser.parse("a, b, c") == ["a", "b", "c"]
    assert parser.parse("a,b,c") == ["a", "b", "c"]


def test_JsonOutputParser():
    parser = JsonOutputParser()
    assert parser.parse('{"a": 1, "b": 2}') == {"a": 1, "b": 2}
    assert parser.parse('```{"a": 1, "b": 2}```') == {"a": 1, "b": 2}
    assert parser.parse('```json{"a": 1, "b": 2}```') == {"a": 1, "b": 2}
    assert parser.parse('{"a": 1, "b": 2}') == {"a": 1, "b": 2}
    assert parser.parse('{"a": 1, "b": 2, "c": [1, 2, 3]}') == {
        "a": 1,
        "b": 2,
        "c": [1, 2, 3],
    }
    assert parser.parse('{"a": 1, "b": 2, "c": [1, 2, 3]}') == {
        "a": 1,
        "b": 2,
        "c": [1, 2, 3],
    }
