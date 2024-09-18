# @param [String] a
# @param [String] b
# @param [Integer] n
def block_concat(a, b, n)
  s = ''
  pos2 = n
  if a.length>b.length
    k = a.length
  else
    k = b.length
  end
  while pos2<=k
    if a!=nil
      if a.length<n
        s = s+a
        if b!=''
          s=s+ b[0]*(n-a.length)
        end
        b = b[n-a.length+1..]
        a=''
      else
        s = s+a[0...n]
        a = a[n..]
      end
    end
    if b!=nil
    if b.length<n
        s = s+b
        if a!=''
          s = s+a[0]*(n-b.length)
        end
        a = a[n-b.length..]
        b=''
      else
        s = s+b[0...n]
        b = b[n...]
    end
    end
    pos2+=n
  end
    return s
end

puts block_concat('aaaaaaa', 'bbbb', 3)