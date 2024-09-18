# Написать функцию для нахождения площади наибольшего треугольника по трем сторонам
# Формула Герона: S=sqrt(p(p-a)(p-b)(p-c)), где p - полупериметр
# Треугольники могут быть переданы в виде массивов [a,b,c] или хэшей { a: 2, b: 6, c: 7 }
class InvalidTriangle < RuntimeError; end

def isTriangleExists(a,b,c)
  if (a!=nil)and(b!=nil)and(c!=nil)and(a>0)and(b>0)and(c>0)and(a+b>c)and(a+c>b)and(b+c>a)
    true
  else
    /p 'errors'
    throw InvalidTriangle/
    false
  end

end

def largest_triangle(triangles)
  s = 0
  max = 0
    triangles.each{
      |triangle|
      if triangle.kind_of?(Array)
        a = triangle[0]
        b = triangle[1]
        c = triangle[2]
      else
        a = triangle[:a]
        b = triangle[:b]
        c = triangle[:c]
      end
      if isTriangleExists(a,b,c)
        p = (a+b+c)/2.0
        s = Math.sqrt(p*(p-a)*(p-b)*(p-c))
        if s>max
          max = s
        end
      else
        throw InvalidTriangle
      end
    }
  /if max==0
      throw InvalidTriangle
    end/
  max
end

#p largest_triangle([[-1, 1, 1], { a: 2, b: -6, c: 7 }, [8, 4, 5]])


