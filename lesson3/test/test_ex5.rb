require_relative '../lib/ex5'
require 'test/unit'

class TestTriangles < Test::Unit::TestCase
  TOLERANCE = 10e-7

  def test_empty_array
    assert_equal(0, largest_triangle([]))
  end

  def test_common_case
    assert_operator (8.1815340859768 - largest_triangle([[1, 1, 1], [2, 6, 7], [8, 4, 5]])).abs,
                    :<,
                    TOLERANCE
  end

  def test_mixed_objects
    assert_operator (8.1815340859768 - largest_triangle([[1, 1, 1],
                                                         { a: 2, b: 6, c: 7 },
                                                         [8, 4, 5]])).abs,
                    :<,
                    TOLERANCE
  end

  def test_invalid_values
    assert_throw(InvalidTriangle) do
      largest_triangle([[-1, 1, 1], [-1, 1, 1], [-1, 1, 1]])
      #largest_triangle([-1,1,1],{ a: 2, b: -6, c: 7 },[8,4,5])
    end
  end
end