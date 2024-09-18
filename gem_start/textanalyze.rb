require "torch"
require "torchtext"
class Counter < Hash
  def initialize(other = nil)
    super(0)
    if other.is_a? Array
      other.each { |e| self[e] += 1 }
    end
    if other.is_a? Hash
      other.each { |k,v| self[k] = v }
    end
    if other.is_a? String
      other.each_char { |e| self[e] += 1 }
    end
  end
  def +(rhs)
    raise TypeError, "cannot add #{rhs.class} to a Counter" if ! rhs.is_a? Counter
    result = Counter.new(self)
    rhs.each { |k, v| result[k] += v }
    result
  end
  def -(rhs)
    raise TypeError, "cannot subtract #{rhs.class} to a Counter" if ! rhs.is_a? Counter
    result = Counter.new(self)
    rhs.each { |k, v| result[k] -= v }
    result
  end
  def most_common(n = nil)
    s = sort_by {|k, v| -v}
    return n ? s.take(n) : s
  end
  def to_s
    "Counter(#{super.to_s})"
  end
  def inspect
    to_s
  end
end
# -----------------------------------------------------------

def make_vocab()
  train_iter, _ = TorchText.Datasets.AG_NEWS()
  counter = Counter()
  train_iter.each { |(label, line)|
    counter.update(tokenizer(line))
    result = TorchText.vocab.Vocab(counter, min_freq = 1)
    lngth = result.length
    return result, lngth
  }
end
  # -----------------------------------------------------------

  tokenizer = TorchText.data.utils.get_tokenizer("basic_english")
  vocab, vocab_size = make_vocab()

  text_pipeline = lambda do |x|{[vocab[token]
  for token in tokenizer(x)]}
  label_pipeline = lambda do |x|
    return x-1
  end


  # -----------------------------------------------------------

  def collate_batch(batch)
    label_list, text_list, offsets = [], [], [0]
     for (_label, _text) in batch
    label_list.append(label_pipeline(_label))
    processed_text = T.tensor(text_pipeline(_text),
                              dtype=Torch.int64)
    text_list.append(processed_text)
    offsets.append(processed_text.size(0))
    label_list = Torch.tensor(label_list, dtype=T.int64)
    offsets = Torch.tensor(offsets[0...-1]).cumsum(dim=0)
    text_list = Torch.cat(text_list)
    return label_list.to(device), text_list.to(device), \
    offsets.to(device)

    # -----------------------------------------------------------

    class TextClassificationModel<Torch.nn.Module
      def __init__( vocab_size, embed_dim, num_class)
        super(TextClassificationModel).__init__()
      @embedding = Torch::NN.EmbeddingBag(vocab_size,
                                         embed_dim, sparse=true)
      @fc = Torch::NN.Linear(embed_dim, num_class)
      init_weights()

      def init_weights()
        lim = 0.5
        @embedding.weight.data.uniform_(-lim, lim)
        @fc.weight.data.uniform_(-lim, lim)
        @fc.bias.data.zero_()

      def forward( text, offsets)
        embedded = @embedding.call(text, offsets)
      return @fc.call(embedded)

      # -----------------------------------------------------------

      def train(model, dataloader, optimizer, criterion, epoch)
        model.train()
      total_acc, total_count = 0, 0
      log_interval = 500
      start_time = time.time()

      for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predited_label = model(text, offsets)
        loss = criterion(predited_label, label)
        loss.backward()
        T.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predited_label.argmax(1) == \
      label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx != 0
          elapsed = time.time() - start_time
        print("help")
        total_acc, total_count = 0, 0
        start_time = time.time()

        # -----------------------------------------------------------

        def evaluate(model, dataloader, criterion):
          model.eval()
        total_acc, total_count = 0, 0

        Torch.no_grad()
          for idx, (label, text, offsets) in enumerate(dataloader)
            predited_label = model(text, offsets)
            loss = criterion(predited_label, label)
            total_acc += (predited_label.argmax(1) == \
        label).sum().item()
            total_count += label.size(0)
            return total_acc/total_count

            # -----------------------------------------------------------

            def predict(model, text, text_pipeline)
              model.eval()
            with T.no_grad()
              text = Torch.tensor(text_pipeline(text))
            output = model(text, T.tensor([0]))
            return output.argmax(1).item() + 1

            # -----------------------------------------------------------

            def main()
              print("\nBegin AG News classification demo \n")

            np.random.seed(1)
            T.manual_seed(1)

            num_classes = 4
            emsize = 64
            model = TextClassificationModel(vocab_size, \
    emsize, num_classes).to(device)

            # Hyperparameters
            EPOCHS = 3
            LR = 5.0         # learning rate
            BATCH_SIZE = 64  # for training

            criterion = T.nn.CrossEntropyLoss()
            optimizer = T.optim.SGD(model.parameters(), lr=LR)
            scheduler = T.optim.lr_scheduler.StepLR(optimizer,
                                                    1.0, gamma=0.1)
            total_accu = None

            train_iter, test_iter = tt.datasets.AG_NEWS()  # reset

            train_dataset = list(train_iter)
            test_dataset = list(test_iter)
            num_train = int(len(train_dataset) * 0.95)
            split_train_, split_valid_ = \
    T.utils.data.dataset.random_split(train_dataset,
                                      [num_train, len(train_dataset) - num_train])

            train_dataloader = T.utils.data.DataLoader(split_train_, \
    batch_size=BATCH_SIZE, shuffle=True,
                                                       collate_fn=collate_batch)
            valid_dataloader = T.utils.data.DataLoader(split_valid_, \
    batch_size=BATCH_SIZE, shuffle=True,
                                                       collate_fn=collate_batch)
            test_dataloader = T.utils.data.DataLoader(test_dataset, \
    batch_size=BATCH_SIZE, shuffle=True,
                                                      collate_fn=collate_batch)

            # -----------------------------------------------------------

            print("Starting training \n")

            for epoch in range(1, EPOCHS + 1):
              epoch_start_time = time.time()
              train(model, train_dataloader, optimizer,
                    criterion, epoch)
              accu_val = evaluate(model, valid_dataloader, criterion)
              # replace "gt" with Boolean operator symbol
              if total_accu is not None and total_accu "gt" accu_val:
                scheduler.step()
              else:
                total_accu = accu_val
              print('-' * 59)
              print('| end of epoch {:3d} | time: {:5.2f}s | '
              'valid accuracy {:8.3f} '.format(epoch,
                                               time.time() - epoch_start_time, accu_val))
              print('-' * 59)

              print("\nDone ")

              # -----------------------------------------------------------

              print("\nComputing model accuracy on test dataset ")
              accu_test = evaluate(model, test_dataloader, criterion)
              print("test accuracy = {:8.3f}".format(accu_test))

              ag_news_label = {1: "World", 2: "Sports", 3: "Business",
                               4: "Sci/Tec"}

              text_str = "Last night the Lakers beat the Rockets by " \
    + "a score of 100-95. John Smith scored 23 points."
              print("\nInput text: ")
              print(text_str)

              c = predict(model, text_str, text_pipeline)
              print("\nPredicted class: " + str(c))
              print(ag_news_label[c])

              print("\nEnd demo \n")

              # -----------------------------------------------------------

              if __name__ == "__main__"
                main()

