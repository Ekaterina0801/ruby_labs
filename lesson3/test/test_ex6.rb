require_relative '../lib/ex6'
require 'test/unit'
class TestDiff < Test::Unit::TestCase

  def test_different_variable
    assert_equal('0', diff('3*x^2+4*x+4', 'z'))
  end

  def test_common_case
    assert_equal('6*x-4', diff('3*x^2-4*x+4', 'x'))
  end

  def test_empty_string
    assert_equal('0', diff('', 'x'))
  end

  def test_invalid_operation
    assert_throw(InvalidOperation) do
      diff('3~x^2-4*x+4', 'x')
    end
  end

end
