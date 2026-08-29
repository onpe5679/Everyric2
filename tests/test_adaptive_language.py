from everyric2.alignment.language import detect_language_from_text


def test_script_language_router() -> None:
    assert detect_language_from_text("hello world") == ("en", False)
    assert detect_language_from_text("노래 가사")[0] == "ko"
    assert detect_language_from_text("歌をうたう")[0] == "ja"
    assert detect_language_from_text("中文歌词")[0] == "zh"
    assert detect_language_from_text("hello world forever 노래") == ("ko", True)
