def maxabspair(a)
  a.sort!{|a,b|a.abs<=>b.abs}.last(2)
end

a = Array.new(10) { rand(-20.0..20.0) }
p a
p maxabspair(a)
