def number_of_point(x,y)
    if (x<0)||(y<0)||(x>28)||(y>28)
        nil
    elsif (x*x+(y-7.5)*(y-7.5)>6.25*6.25)||(x-28)*(x-28)+(y-7.5)*(y-7.5)>6.25*6.25
        3
    else 
        2
    end
end            

p 'Проверка'
p number_of_point(20,3)
p number_of_point(-2,10)
p number_of_point(10,10)
