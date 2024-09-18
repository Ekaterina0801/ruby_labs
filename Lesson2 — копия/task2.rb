def prodcomplex(a,b,c,d)
  x1 = Complex(a,b)
  x2 = Complex(c,d)
  r = x1*x2
  [r.real,r.imaginary]
end
p 'Проверка'
p prodcomplex(0,0,0,0)
p prodcomplex(1,2,3,4)
p prodcomplex(1,5,6,7)