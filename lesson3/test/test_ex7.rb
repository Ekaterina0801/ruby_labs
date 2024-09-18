require_relative '../lib/ex7'
require 'test/unit'
class TestRank < Test::Unit::TestCase

  def test_common_case
    assert_equal(["LAL","MIA","CLE","SAC"],rank_teams([{ home: 'LAL', away: 'CLE', score: '120:115' }, { home: 'MIA', away: 'SAC', score: '111:102' },
                    { home: 'SAC', away: 'LAL', score: '98:114' }, { home: 'CLE', away: 'MIA', score: '100:96' }]))
  end

  def test_one_match
    assert_equal(["LAL","MIA"], rank_teams([{home:"MIA",away:"LAL",score:"100:105"}]))
  end

  def test_invalid_rank
    assert_throw(InvalidRes) do
      rank_teams([{home:"MIA",away:"LAL",score:"100:105"},{home:"LAL",away:"MIA",score:"100:105"}])
    end
  end

end

