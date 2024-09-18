# @param [String] s
def sequence(s)
  if s == ''
    return 0
  end
  cc = 1 #текущий счетчик
  m = 0 #max
  s.downcase!
  s0 = ''
  s.each_char do |k| 
    if k == s0
      cc+=1
    else
      if cc > m
        m = cc
      end
    cc = 1
    end
    s0 = k
  end
  return m
end

