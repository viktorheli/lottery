--Uncoment cuda if you have cuda :)
require ('paths')
require 'io'
require 'nn'
require 'optim'
--require 'cunn'
--require 'cutorch'

torch.setdefaulttensortype('torch.FloatTensor')

--cmd line arg
cmd = torch.CmdLine()
cmd:text()
cmd:text('Make train for lottery')
cmd:text('Example:')
cmd:text('$> th train.lua -dataset "path to dataset" -storenet "path to stored neuralnet" -learningrate "number" -saveevery "temporal save every N epochs"')
cmd:text('Options:')
cmd:option('-dataset', 'lottery.t7', 'Path to load dataset')
cmd:option('-storenet', 'lottery.dat', 'Path to saving neuralnet')
cmd:option('-train', '40000', 'Numbers of train iterations')
cmd:option('-learningrate', '0.0001', 'Learning rate')
cmd:option('-saveevery', '5000', 'Save temporal net every N "epoch\'s"')
cmd:option('-valid', '1000', 'Do validation on dataset every N epochs and display min max and average error')
cmd:option('-progress', 'yes', 'Display progress bar "yes" or "no"')
cmd:option('-normalizing', '100', 'Normalizing value')

opt = cmd:parse(arg or {})

normalized = opt.normalizing
dataset = torch.load(opt.dataset)
inpu = torch.Tensor(dataset.input)--:cuda()
outpu = torch.Tensor(dataset.output)--:cuda()                                                                                                         
                                                                                                                                                      
function validation() --This is not real validation                                                                                                   
                                                                                                                                                      
        dsize = inpu:size(1)                                                                                                                          
        errormatrix = {}                                                                                                                              
                                                                                                                                                      
        for i = 1, dsize/10 do                                                                                                                        
                                                                                                                                                      
                permutation = torch.random(dsize)                                                                                                     
                                                                                                                                                      
                fwd = mlp:forward(inpu[permutation])--:cuda()                                                                                         
                predict = (fwd*normalized)                                                                                                            
                real = (outpu[permutation]*normalized)                                                                                                
                erorrpercent = math.abs((((predict[1]/real[1])-1)*100))                                                                               
                                                                                                                                                      
                table.insert(errormatrix, erorrpercent)                                                                                               
                                                                                                                                                      
        end                                                                                                                                           
        avger = torch.min(torch.Tensor(errormatrix))                                                                                                  
                                                                                                                                                      
        print("Error, %: "..avger)                                                                                                                    
                                                                                                                                                      
end                                                                                                                                                   
                                                                                                                                                      
if (paths.filep(opt.storenet) == true) then
        
                print("\n".."++++++++++++++++++Loading net file:        "..opt.storenet.."  ++++++++++++++++++++".."\n")
                mlp = torch.load(opt.storenet)
                print (mlp)
                
        else 
                print("\n".."++++++++++ WARNING!!! Creating BIG MLP for little volume of data!!!+++++++++++++".."\n")
                mlp = nn.Sequential()
                mlp:add(nn.Linear(45, 3096))
                mlp:add(nn.Sigmoid())
                mlp:add(nn.Linear(3096, 45))
                mlp:add(nn.Sigmoid())
--                mlp:cuda()
                print(mlp)
        
end --this end for if for mlp


criterion = nn.MSECriterion()--:cuda()
params, gradParams = mlp:getParameters()
optimState = {learningRate = opt.learningrate}

for epoch = 1, opt.train do
        if (opt.progress == "yes" ) then
                xlua.progress(epoch, opt.train)
        end
        function feval(params)
                rndidx = torch.random(outpu:size(1))
                gradParams:zero()
                local outputs = mlp:forward(inpu)
                local loss = criterion:forward(outputs, outpu)
                local dloss_doutputs = criterion:backward(outputs, outpu)

                mlp:backward(inpu, dloss_doutputs)

                return loss, gradParams
        end

        fs = optim.adam(feval, params, optimState)

        if  epoch % opt.saveevery  == 0 then
                epochloss = fs[1] / outpu:size(1)
                --fwd = mlp:forward(dataset.input[rndidx])

                print("Number of iteration: "..epoch.." of "..opt.train)
                print("Epochloss:       "..epochloss)
--              validation()
                print("Saving tempotary model to: "..opt.storenet.."temporal")
                torch.save(opt.storenet.."temporal", mlp)

        end

        if epoch % opt.valid  == 0 then
                print("Number of iteration: "..epoch.." of "..opt.train)
                validation()
                epochloss = fs[1] / outpu:size(1)
                print("Epochloss:       "..epochloss)
        end

end
print("Saving model to: "..opt.storenet)
torch.save(opt.storenet, mlp)
validation()
