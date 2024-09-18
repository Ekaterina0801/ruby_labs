class Classifier
  attr_writer :batch_size, :count_epochs
  def initialize(batch_size=64, count_epochs=5)
    super()
    @batch_size = batch_size
    @count_epochs  = count_epochs
  end
  def print
    puts @batch_size, @count_epochs
  end
end

n = Classifier.new(batch_size=5)
#n = n.call(batch_size:56)
n.print