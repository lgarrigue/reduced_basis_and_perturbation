# Module which enables to manage recursive quantities computation
px = println

mutable struct Stock
    Xs # quantities to be recursively computed
    kinds # there are length(X) different recursive quantities
    computation_functions
    p # parameters
    debug
    function Stock(kinds,computation_functions,parameters;debug=false)
        St = new()
        St.debug = debug
        # First stuff
        St.kinds = kinds
        St.p = parameters
        # X
        St.Xs = Dict()
        for k in St.kinds
            St.Xs[k] = Dict()
        end
        # Computation functions
        St.computation_functions = Dict()
        for i in 1:length(St.kinds)
            k = St.kinds[i]
            St.computation_functions[k] = computation_functions[i]
        end
        St
    end
end

# Needs a value, computes it if it was not computed previously
function X(kind,coords,St::Stock)
    if !haskey(St.Xs[kind],coords)
        if St.debug
            px("Computes ",kind," ",coords)
        end
        x = St.computation_functions[kind](coords,St)
        St.Xs[kind][coords] = x
    end
    St.Xs[kind][coords]
end

function sets_value(x,kind,coords,St::Stock)
    if haskey(St.Xs[kind],coords)
        print("FOR TYPE ",kind," CANNOT SET VALUE ",x," WITH COORDS ",coords," because it is already set")
    end
    St.Xs[kind][coords] = x
end

# gives all the elements of kind kind
all_elements(kind,St::Stock) = [v for v in S.Xs[kind]]
