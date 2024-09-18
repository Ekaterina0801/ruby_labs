gem "torchvision"
require "torch"
require "torchvision"
require "tensorflow"
require "torchtext"
ngrams = 2

#train_dataset, test_dataset = TorchText::Datasets::AG_NEWS.load(root: ".data", ngrams: ngrams)
base_csv = '/Users/ekaterinaaleksandrovna/Downloads/IMDB Dataset.csv'
df = CSV.read(base_csv)
#X,y = df['review'].values,df['sentiment'].values
ind = 1
train_dataset = Array.new(df.size-1)
test_dataset = Array.new(df.size-1)
(1...df.size - 1).each { |i|
  train_dataset[ind - 1] = df[ind][0]
  test_dataset[ind - 1] = df[ind][1]
  ind += 1
}
x_train,x_test = train_test_split(test_dataset)
y_train,y_test = train_test_split(train_dataset)
device = Torch.device(Torch::CUDA.available? ? "cuda" : "cpu")
#-----------------------Parameters----------------------------
vocab_size = train_dataset.vocab.length
batch_size = 16
embed_dim = 64
num_class = 4
lr = 0.4
gamma = 0.9
step_size_for_optim = 1
labels =        {1 =>  "World",
                 2 => "Sports",
                 3 => "Business",
                 4 => "Sci/Tec"}
n_epochs = 5
min_valid_loss = Float::INFINITY
#--------------------------------------------------------------

class TextClassifier < Torch::NN::Module
  def initialize(vocab_size=1308843, embed_dim=32, num_class=4)
    super()
    @embedding = Torch::NN::EmbeddingBag.new(vocab_size, embed_dim, sparse: true)
    @fc1 = Torch::NN::Linear.new(embed_dim, 4)
    set_weights
  end


  def set_weights
    lim = 0.5
    @embedding.weight.data.uniform!(-lim, lim)
    @fc1.weight.data.uniform!(-lim, lim)
    @fc1.bias.data.zero!
  end

  def forward(text, offsets)
    @fc1.call(@embedding.call(text, offsets:offsets))
  end
end


model = TextClassifier.new(vocab_size, embed_dim, num_class).to(device)
#new_model = TestClass.new().to(device)
criterion = Torch::NN::CrossEntropyLoss.new.to(device)
optimizer = Torch::Optim::SGD.new(model.parameters, lr: lr)
scheduler = Torch::Optim::LRScheduler::StepLR.new(optimizer, step_size: step_size_for_optim, gamma: gamma)

generate_batch = lambda do |batch|
  label = Torch.tensor(batch.map { |entry| entry[0] })
  text = batch.map { |entry| entry[1] }
  offsets = [0] + text.map { |entry| entry.size }

  offsets = Torch.tensor(offsets[0..-2]).cumsum(0)
  text = Torch.cat(text)
  [text, offsets, label]
end

train_do = lambda do |sub_train_|
  # Train the model
  train_loss = 0
  train_acc = 0
  data = Torch::Utils::Data::DataLoader.new(sub_train_, batch_size: batch_size, shuffle: true, collate_fn: generate_batch)
  data.each_with_index do |(text, offsets, cls), i|
    optimizer.zero_grad
    text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
    output = model.call(text, offsets)
    loss = criterion.call(output, cls)
    train_loss += loss.item
    loss.backward
    optimizer.step
    train_acc += output.argmax(1).eq(cls).sum.item
  end

  # Adjust the learning rate
  scheduler.step

  [train_loss / sub_train_.length, train_acc / sub_train_.length.to_f]
end

test_do = lambda do |data_|
  loss = 0
  acc = 0
  data = Torch::Utils::Data::DataLoader.new(data_, batch_size: batch_size, collate_fn: generate_batch)
  data.each do |text, offsets, cls|
    text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
    Torch.no_grad do
      output = model.call(text, offsets)
      loss = criterion.call(output, cls)
      loss += loss.item
      acc += output.argmax(1).eq(cls).sum.item
    end
  end

  [loss / data_.length, acc / data_.length.to_f]
end



train_len = (train_dataset.length * 0.95).to_i
sub_train_, sub_test_ = Torch::Utils::Data.random_split(train_dataset, [train_len, train_dataset.length - train_len])

(1...n_epochs).each do |epoch|
  start_time = Time.now
  train_loss, train_acc = train_do.call(sub_train_)
  test_loss, test_acc = test_do.call(sub_test_)

  secs = Time.now - start_time
  mins = secs / 60
  secs = secs % 60

  puts "Эпоха: %d | время:  %d минут, %d секунд" % [epoch + 1, mins, secs]
  puts "\tПотери (loss): %.4f (train)\t|\tТочность (accuracy): %.1f%% (train)" % [train_loss, train_acc * 100]
  puts "\tПотери (loss): %.4f (test)\t|\tТочность (accuracy): %.1f%% (test)" % [test_loss, test_acc * 100]
end

puts "Checking the results of test dataset..."
test_loss, test_acc = test_do.call(test_dataset)
puts "\tLoss: %.4f (test)\t|\tAcc: %.1f%% (test)" % [test_loss, test_acc * 100]



def predict(text, model, vocab, ngrams)
  tokenizer = TorchText::Data::Utils.tokenizer("basic_english")
  Torch.no_grad do
    text = Torch.tensor(TorchText::Data::Utils.ngrams_iterator(tokenizer.call(text), ngrams).map { |token| vocab[token] })
    model.call(text, Torch.tensor([0])).argmax(1).item + 1
  end
end

ex_text_str = <<~EOS
  Good movie
EOS

vocab = train_dataset.vocab
model = model.to("cpu")

puts "This is a %s news" % labels[predict(ex_text_str, model, vocab, 2)]


