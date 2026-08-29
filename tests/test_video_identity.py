"""video_id 기반 서버측 YouTube 제목/채널 복구 — 네트워크 없는 단위 테스트."""

from everyric2.server.services import video_identity as vi


class _Response:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


def test_fetch_video_identity_reads_and_caches_oembed(monkeypatch):
    vi._cache.clear()
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return _Response({"title": "POLARIS - Patterns ft. @rino", "author_name": "Patterns"})

    monkeypatch.setattr(vi.requests, "get", fake_get)
    first = vi.fetch_video_identity("bEV_tH_yrIc")
    second = vi.fetch_video_identity("bEV_tH_yrIc")
    assert first == second == vi.VideoIdentity(
        title="POLARIS - Patterns ft. @rino", channel="Patterns"
    )
    assert len(calls) == 1
    assert calls[0][1]["timeout"] == 2.0


def test_invalid_video_id_never_calls_youtube(monkeypatch):
    monkeypatch.setattr(
        vi.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network called")),
    )
    assert vi.fetch_video_identity("../not-video") is None


def test_channel_resolves_both_hyphen_orientations():
    reversed_identity = vi.VideoIdentity(
        title="POLARIS - Patterns ft. @rino", channel="Patterns"
    )
    standard_identity = vi.VideoIdentity(title="IU - Blueming", channel="IU")
    assert vi.ordered_title_candidates(reversed_identity) == ["POLARIS"]
    assert vi.ordered_title_candidates(standard_identity) == ["Blueming"]


def test_unresolved_title_keeps_the_full_public_title():
    identity = vi.VideoIdentity(title="Track A / Track B", channel="Unrelated Channel")
    assert vi.ordered_title_candidates(identity) == ["Track A / Track B"]
