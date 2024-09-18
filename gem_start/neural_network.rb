require 'torch-rb'
require 'torchtext'
class Classifier
  attr_writer :batch_size, :count_epochs
  attr_reader :train_dataset, :test_dataset
  @train_dataset, @test_dataset = TorchText::Datasets::AG_NEWS.load(root: ".data", ngrams: 2)
  class TextClassifier < Torch::NN::Module


    def initialize(vocab_size, embedding_dimension, num_class)
      super()
      lim = 0.5
      @embedding = Torch::NN::EmbeddingBag.new(vocab_size, embedding_dimension, sparse: true)
      @fc = Torch::NN::Linear.new(embedding_dimension, num_class)
      @embedding.weight.data.uniform!(-lim, lim)
      @fc.weight.data.uniform!(-lim, lim)
      @fc.bias.data.zero!
    end

    def forward(text, offsets)
      @fc.call(@embedding.call(text, offsets: offsets))
    end
  end

  def initialize(batch_size=64, count_epochs=5)
    super()
    @batch_size = batch_size
    @count_epochs  = count_epochs
  end

  def createModel

    #-----------------------Parameters----------------------------
    batch_size = @batch_size # batch size for training
    count_epochs = @count_epochs
    learning_rate = 5
    gamma_for_sheduler = 0.9
    embedding_dimension = 32
    #-------------------------------------------------------------

    labels = {1 => "World",2 => "Sports",3 => "Business",4 => "Sci/Tec"}

    #----------------------Preparing dataset-------------------------
    puts "Preparing dataset"
    #train_dataset, test_dataset = TorchText::Datasets::AG_NEWS.load(root: ".data", ngrams: 2)
    train_len = (@train_dataset.length * 0.8).to_i
    split_train, split_test = Torch::Utils::Data.random_split(@train_dataset, [train_len, @train_dataset.length - train_len])
    device = Torch.device(Torch::CUDA.available? ? "cuda" : "cpu")
    #----------------------------------------------------------------

    #-----------------------Definition network-----------------------

    #----------------------------------------------------------------
    vocab_size = @train_dataset.vocab.length
    count_of_classes = @train_dataset.labels.length
    model = TextClassifier.new(vocab_size, embedding_dimension, count_of_classes).to(device)

    criterion = Torch::NN::CrossEntropyLoss.new.to(device)
    optimizer = Torch::Optim::SGD.new(model.parameters, lr: learning_rate)
    scheduler = Torch::Optim::LRScheduler::StepLR.new(optimizer, step_size: 1, gamma: gamma_for_sheduler)

    collate_batch = Proc.new{
      |batch|
      label = Torch.tensor(batch.map { |entry| entry[0] })
      text = batch.map { |entry| entry[1] }
      offsets = [0] + text.map { |entry| entry.size }
      offsets = Torch.tensor(offsets[0..-2]).cumsum(0)
      text = Torch.cat(text)
      [text, offsets, label]
    }

    #----------------------Training network----------------------
    train = Proc.new{
      |split_train|
      data_train = Torch::Utils::Data::DataLoader.new(split_train, batch_size: batch_size, shuffle: true, collate_fn: collate_batch)
      train_loss,train_accuracy = 0,0
      data_train.each_with_index {
        |(text, offsets, label), i|
        optimizer.zero_grad
        text = text.to(device)
        offsets = offsets.to(device)
        label = label.to(device)
        predicted_label = model.call(text, offsets)
        loss = criterion.call(predicted_label, label)
        train_loss += loss.item
        loss.backward
        optimizer.step
        prediction = predicted_label.argmax(1)
        are_eq = prediction.eq(label)
        train_accuracy += are_eq.sum.item
      }
      scheduler.step
      [train_loss / split_train.length, train_accuracy / split_train.length.to_f]
    }
    #--------------------------------------------------------------

    #----------------------Evaluating network----------------------
    evaluate = Proc.new{
      |split_test|
      data_test = Torch::Utils::Data::DataLoader.new(split_test, batch_size: batch_size, collate_fn: collate_batch)
      loss,accuracy = 0,0
      data_test.each do |text, offsets, label|
        text = text.to(device)
        offsets = offsets.to(device)
        label = label.to(device)
        Torch.no_grad do
          predicted_label = model.call(text, offsets)
          loss = criterion.call(predicted_label, label)
          loss += loss.item
          accuracy += predicted_label.argmax(1).eq(label).sum.item
        end
      end
      [loss / split_test.length, accuracy / split_test.length.to_f]
    }
    #--------------------------------------------------------------



    (1..count_epochs).each {
      |epoch|
      start_time = Time.now
      train_loss, train_accuracy = train.call(split_train)
      test_loss, test_accuracy = evaluate.call(split_test)
      final_time = Time.now
      time =  final_time - start_time
      puts "Epoch: %d | time:  %d seconds" % [epoch, time]
      puts "\tTrain Loss: %.4f \t|\tTrain Accuracy: %.1f%% " % [train_loss, train_accuracy * 100]
      puts "\tTest  Loss: %.4f \t|\tTest  Accuracy: %.1f%% " % [test_loss, test_accuracy * 100]
      puts("-------------------------------------------------")
    }

    puts "Checking the results of test dataset"
    test_loss, test_accuracy = evaluate.call(@test_dataset)
    puts "Test accuracy: %.1f%% " % [test_accuracy * 100]
    Torch.save(model.state_dict, "agnewsNet.pth")
    model
  end


  def make_prediction(text)
    embedding_dimension = 32
    count_of_classes = 4
    device = Torch.device(Torch::CUDA.available? ? "cuda" : "cpu")
    model = TextClassifier.new(@train_dataset.vocab.length, embedding_dimension, count_of_classes).to(device)
    model.load_state_dict(Torch.load("net.pth"))
    model.eval
    prediction = 1
    Torch.no_grad do
      text = Torch.tensor(TorchText::Data::Utils.ngrams_iterator(TorchText::Data::Utils.tokenizer("basic_english").call(text), ngrams).map { |token| vocab[token] })
      prediction = model.call(text, Torch.tensor([0])).argmax(1).item + 1
    end
    puts "Theme of news: %s" % labels[prediction]
  end
end

v = Classifier.new
v.createModel()


