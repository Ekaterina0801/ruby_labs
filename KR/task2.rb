def isNumber(element)
  element.to_f.to_s==element.to_s||element.to_i.to_s==element.to_s
end

def sumNumbersInStr(str)
  sum = 0
  symbols = str.split(';')
  symbols.each do
    |symbol|
    if isNumber(symbol)
      sum+=symbol.to_i
    end
  end
  sum
end

require 'test/unit'

class SumNumbersTest<Test::Unit::TestCase
  TOLERANCE = 10e-7
  def test_empty_str
    assert_equal(0,sumNumbersInStr(''))
  end

  def test_empty_str2
    assert_equal(0,sumNumbersInStr('a;b;c;d;f'))
  end
  def test_common_1
    assert_equal(59,sumNumbersInStr('4;word;55;hello'))
  end

  def test_common_common2
    assert_equal(121,sumNumbersInStr('40;word;55;hello;5;6;7;8;a'))
  end
end
