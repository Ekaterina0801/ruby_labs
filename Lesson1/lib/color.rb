# @param [Integer] r
# @param [Integer] g
# @param [Integer] b
def color(r, g, b)
  # 100.to_s(16) => 64
  dec_to_hex(r)+dec_to_hex(g)+dec_to_hex(b)
end

def dec_to_hex(a)
  if a<0
   '00'
  elsif a<16
    '0'+a.to_s(16).upcase
  elsif a<256
    a.to_s(16).upcase
  else
    'FF'
  end
end

