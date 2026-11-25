from html.parser import HTMLParser

from fireguard.video_sync import synchronized_video_html


class VideoParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.videos = []

    def handle_starttag(self, tag, attrs):
        if tag == "video":
            self.videos.append(dict(attrs))


def test_result_is_the_only_player_with_controls():
    html = synchronized_video_html(b"result-video", b"steps-video")
    parser = VideoParser()
    parser.feed(html)

    assert len(parser.videos) == 2
    assert parser.videos[0]["id"] == "result"
    assert "controls" in parser.videos[0]
    assert parser.videos[1]["id"] == "steps"
    assert "controls" not in parser.videos[1]
    assert "muted" in parser.videos[1]
    assert 'videoUrl("cmVzdWx0LXZpZGVv")' in html
    assert 'videoUrl("c3RlcHMtdmlkZW8=")' in html
