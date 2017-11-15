
torch.setdefaulttensortype('torch.FloatTensor')

--cmd line arg
cmd = torch.CmdLine()
cmd:text()
cmd:text('Making train and test datasets')
cmd:text('Example:')
cmd:text('$> th make-dataset.lua  -csvpath data6from45.csv')
cmd:text('Options:')

cmd:option('-csvpath', 'data6from45.csv', 'Path to csv file with data')
cmd:option('-startidx', '1', 'Start from index')
cmd:option('-endidx', '3000', 'End index')
cmd:option('-norm', '100', 'Normalizing value')
cmd:option('-save', 'lottery.t7', 'Path to save dataset')
opt = cmd:parse(arg or {})



delimiter = ","
csv_path = opt.csvpath
X_table = {}
start_idx = tonumber(opt.startidx)
batch_size = opt.endidx
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

input = {}
ssi = {}
sso = {}
for i = 1, X_tensor:size(1) do

                for b = 1, X_tensor:size(2) do
--                      if X_tensor[i][b] > 0 then
--                              ssi[b] = 1
                        ssi[b] = (X_tensor[i][b]/mean)
--                      else
--                              ssi[b] = 0
--                      end
                        ss1 = torch.Tensor(ssi)
                end

        table.insert(input, ss1)

end

inputs = {}
outputs = {}
count = 1
count1 = 1
for i = 1, #input do
        if i%2 == 1 then
                outputs[count1] = input[i]
                count1 = count1 + 1

        else
                inputs[count] = input[i]
                count = count + 1

        end


end

input = torch.Tensor(#inputs, 45)
output = torch.Tensor(#outputs, 45)

for i = 1, #inputs do
        input[i] = inputs[i]
        output[i] = outputs[i]
end


datasave = {input = input, output = output}
print ("Storing training data in file:",opt.save)
torch.save(opt.save, datasave)
dataset = torch.load(opt.save)
print(dataset)

