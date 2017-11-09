--Uncoment cuda if you have cuda :)
require ('paths')
require 'io'
require 'nn'
--require 'cunn'
--require 'cutorch'

torch.setdefaulttensortype('torch.FloatTensor')

--cmd line arg
cmd = torch.CmdLine()
cmd:text()
cmd:text('Making lottery predict')
cmd:text('Example:')
cmd:text('$> th predict.lua')
cmd:text('Options:')
cmd:option('-storenet', 'lottery.dat', 'Path to neuralnet')
cmd:option('-file', '6from45input.csv', 'CSV file with input data')
cmd:option('-normalizing', '100', 'Normalizing value')

opt = cmd:parse(arg or {})
normalized = opt.normalizing

function round(num, numDecimalPlaces)
  return tonumber(string.format("%." .. (numDecimalPlaces or 0) .. "f", num))
end

delimiter = ","
csv_path = opt.file
X_table = {}
start_idx = 1
batch_size = 1
mean = opt.norm

X_table = {}
  local count = 1
  local batch_count = 1
for line in io.lines(csv_path) do
        if count >= start_idx and count<(start_idx + batch_size) then
        X_table[batch_count] = line
        batch_count = batch_count + 1
        end
    count = count + 1
end

num_instance = #X_table[1]:split(delimiter)
X_tensor = torch.zeros(#X_table,num_instance)

for i = 1,#X_table do

        X_tensor[i] = torch.Tensor(X_table[i]:split(delimiter))

end
--print(X_tensor)
input = {}
for i = 1, X_tensor:size(2) do
--      if X_tensor[1][i] > 0 then
--              input[i] = 1 
        input[i] = (X_tensor[1][i]/normalized)
--      else 
--              input[i] = 0 
--      end
end
        inpu = torch.Tensor(input)--:cuda()
--      print(inpu)


mlp = torch.load(opt.storenet)

predict = mlp:forward(inpu)
print(predict)

print("Номер :".."      Значение")

for i = 1, predict:size(1) do

        norm = round(predict[i]*normalized)
        print("Номер :"..i.."  "..norm)
end
