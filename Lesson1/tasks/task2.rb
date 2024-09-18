# Написать в файле block_concat.rb функцию для сложения строк по следующим правилам:
# на вход подается две строки и размер блока. Блоки строк поочередно добавляются к итоговой строке.
# Если одна из строк не содержит больше блоков, то копируется остаток другой строки.
# Последний блок может быть меньшего размера.

# DO NOT MODIFY CODE BELOW
require_relative '../lib/block_concat'
require 'test/unit'

class TestBlockConcat < Test::Unit::TestCase
  def test_whole_blocks
    assert_equal 'aaabbbaaabbb', block_concat('aaaaaa', 'bbbbbb', 3), 'Should concat whole blocks'
  end

  def test_small_blocks
    assert_equal 'aaabbaaa', block_concat('aaaaaa', 'bb', 3), 'Should concat smaller blocks'
  end

  def test_empty
    assert_equal '', block_concat('', '', 5), 'Should handle empty strings'
  end

end