gem "torchvision"
require "torch"
require "torchvision"
require "tensorflow"

class MyNet < Torch::NN::Module
  def initialize
    super()
    '''
    @conv1 = Torch::NN::Conv2d.new(1,32,3,stride:1)
    @conv2 = Torch::NN::Conv2d.new(32,64,3,stride:1)
    @dp1 = Torch::NN::Dropout2d.new(p:0.25)
    @dp2 = Torch::NN::Dropout2d.new(p:0.25)
    @fc1 = Torch::NN::Linear.new(9216,128)
    @fc2 = Torch::NN::Linear.new(128,10)
    '''
    @conv1 = Torch::NN::Conv2d.new(1,32,3,stride:1)
    @conv2 = Torch::NN::Conv2d.new(32,64,3,stride:1)
    @conv3 = Torch::NN::Conv2d.new(64,128,3,stride:1)
    @dp1 = Torch::NN::Dropout2d.new(p:0.25)
    @dp2 = Torch::NN::Dropout2d.new(p:0.5)
    @fc1 = Torch::NN::Linear.new(15488,128)
    @fc3 = Torch::NN::Linear.new(128,84)
    @fc4 = Torch::NN::Linear.new(84,10)
  end

  def forward(x)
    x = @conv1.call(x)
    x = Torch::NN::F.relu(x)
    x = @conv2.call(x)
    x = Torch::NN::F.relu(x)
    x = @conv3.call(x)
    x = Torch::NN::F.relu(x)
    x = Torch::NN::F.max_pool2d(x, 2)
    x = @dp1.call(x)
    x = Torch.flatten(x, start_dim: 1)
    x = @fc1.call(x)
    x = Torch::NN::F.relu(x)
    x = @dp2.call(x)
    x = @fc3.call(x)
    x = Torch::NN::F.relu(x)
    x = @fc4.call(x)
    output = Torch::NN::F.log_softmax(x, 1)
    output
  end
end

class_mapping = [
  "0",
  "1",
  "2",
  "3",
  "4",
  "5",
  "6",
  "7",
  "8",
  "9"
]

def download_mnist_datasets()
  train_data = TorchVision::Datasets::MNIST.new(
    "data",
    train:true,
    download:true,
    transform: TorchVision::Transforms::Compose.new([
                                                      TorchVision::Transforms::ToTensor.new,
                                                      TorchVision::Transforms::Normalize.new([0.1307], [0.3081]),
                                                    ])
    )
validation_data = TorchVision::Datasets::MNIST.new(
  File.join(__dir__, "data"),
  train:false,
  download:true,
  transform:TorchVision::Transforms::Compose.new([
                                                              TorchVision::Transforms::ToTensor.new,
                                                              TorchVision::Transforms::Normalize.new([0.1307], [0.3081]),
                                                            ])
  )
  return train_data, validation_data
end

vv = TensorFlow::Keras::Preprocessing::Image.load_img("/Users/ekaterinaaleksandrovna/Documents/test_image.png")
arr = TensorFlow::Keras::Preprocessing::Image.img_to_array(vv)
def predict(model, input, target, class_mapping)
    model.eval
    Torch.no_grad do
    predictions = model.call(input)
    # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
    predicted_index = predictions[0].argmax(0).item
    predicted = class_mapping[predicted_index]
    expected = class_mapping[target.item]
    end
    return predicted, expected
end

# make an inference

#print(format"Predicted: '{predicted}', expected: '{expected}'")
root = File.join(__dir__, "data")
use_cuda = Torch::CUDA.available?
device = Torch.device(use_cuda ? "cuda" : "cpu")
test_dataset = TorchVision::Datasets::MNIST.new(root,
                                                train: false,
                                                download: true,
                                                transform: TorchVision::Transforms::Compose.new([
                                                                                                  TorchVision::Transforms::ToTensor.new,
                                                                                                  TorchVision::Transforms::Normalize.new([0.1307], [0.3081]),
                                                                                                ])
)

new_model = MyNet.new
new_model.load_state_dict(Torch.load("/Users/ekaterinaaleksandrovna/Documents/mnist.pth"))
validation_data = Torch::Utils::Data::DataLoader.new(test_dataset, batch_size: 64, shuffle: true)
#train_loader = Torch::Utils::Data::DataLoader.new(test_dataset, batch_size: batch_size, shuffle: true)
#input, target = validation_data.dataset[0], validation_data.dataset[0][1]
h = Hash.new
validation_data.each do |data, target|
  predicted, expected = predict(new_model, data, target[1],
                                class_mapping)
  h[:data]=target[0]
  p predicted
  p expected
  print(format"Predicted: '{predicted}', expected: '{expected}'")
  output = new_model.call(data)
  pred = output.argmax(1, keepdim: true)
  p(pred)
  p(target)
end
predicted, expected = predict(new_model, input[0], target,
                              class_mapping)
#print(format"Predicted: '{predicted}', expected: '{expected}'")
# make an inference
predicted, expected = predict(new_model, input, target,
                                class_mapping)
print(format"Predicted: '{predicted}', expected: '{expected}'")
print('PREDICTION')





