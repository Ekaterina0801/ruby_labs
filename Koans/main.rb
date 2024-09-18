class OG
  @title = ''
  @cnt = 0
  def initialize(t, c)
    @title = t
    cnt = c
  end

  def title
    @title
  end

end

pp = OG.new("yyy",3)
i = []
i.push(OG.new("yyy",3))
i.each do |cg|
  p cg.title
end

@courses = []
@enrollments = []
@courses_groups = []
@cg = {}
@enrollments.each do |e|
  @title = ''
  @courses.each do |c|
    if true
      @title = c.title
      break
    end
  end
  if @cg.has_key?(@title)
    @cg[@title] += 1
  else
    @cg[@title] = 1
  end
end
@cg.each { |k, v| @courses_groups.push(CourseGroup.new(k, v)) }
# [{title: 'A', name: 3},...]

def final; end