def maxSquare(circles)
  max_square = 0;
  circles.each do
  |circle|
    if circle[:unit]=='m'
      square = Math::PI*circle[:r]*circle[:r]
      if square>max_square
        max_square = square
      end
    else
      new_r = circle[:r]/100.0
      square = Math::PI*new_r*new_r
      if square>max_square
        max_square = square
      end
    end
  end
  max_square
end

require 'test/unit'

class MaxSquareTest<Test::Unit::TestCase
  TOLERANCE = 10e-7
  def test_empty_hash
    assert_equal(0,maxSquare([]))
  end

  def test_common_hash_1
    assert_operator (12.566370614359172 - maxSquare([{r: 2, unit: 'm'}, {r: 4, unit:'cm'}])).abs,
                    :<,
                    TOLERANCE
  end

  def test_common_hash_2
    assert_operator (78.53981633974483 - maxSquare([{r: 4, unit: 'm'}, {r: 8, unit:'cm'},{r:10,unit:'cm'},{r:5,unit:'m'}])).abs,
                    :<,
                    TOLERANCE
  end

  def test_common_hash_3
    assert_operator (201.06192982974676 - maxSquare([{r: 400, unit: 'сm'}, {r: 800, unit:'cm'},{r:100,unit:'cm'},{r:5,unit:'m'}])).abs,
                    :<,
                    TOLERANCE
  end

end

p maxSquare([{r: 400, unit: 'сm'}, {r: 800, unit:'cm'},{r:100,unit:'cm'},{r:5,unit:'m'}])