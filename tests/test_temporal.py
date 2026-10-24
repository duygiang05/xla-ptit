from fireguard.config import Config
from fireguard.shape import Region
from fireguard.temporal import TemporalFilter

def test_temporal_voting_needs_history():
    f=TemporalFilter(Config(vote_window=5,vote_hits=3))
    r=Region('fire',(5,5,20,20),400,(15,15))
    assert not f.update([r],0)
    assert not f.update([r],1)
    assert len(f.update([r],2))==1
