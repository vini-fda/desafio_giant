using LinearAlgebra, SparseArrays, Combinatorics

# Probability that A(ability x) loses against B(ability y)
# therefore, there will be a probability p^2 
# that B will become the new Mario Kart Master
prob_lose(x, y) = y / (x+y)

struct MarkovState{T<:Integer}
  # Stores the player sequence
  # in which the first element is the Mario Kart Master,
  # the second element is the challenger(first one in the queue)
  # and so on...
  player_sequence::Vector{T}
  # The respective abilities of each player
  ability_sequence::Vector{T}
  # The current amount of times the Mario Kart Master has won
  w::T

  function MarkovState{T}(player_sequence::Vector{T}, H::Vector{T}, w::T) where T <: Integer
      new(player_sequence, H[player_sequence], w)
  end
end

MarkovState(player_sequence, H, w::T) where T <: Integer = MarkovState{T}(player_sequence, H, w)

# generates all possible markov states for a given n
function markov_states(n, H)
  # initial player sequence, e.g. abcd or 1234
  canonical_sequence = 1:n
  # all permutations of player sequences
  p_seqs = permutations(canonical_sequence)
  # amount of times the Mario Kart Master has won
  w_seq = 0:(n-2)

  vec([MarkovState(p_seq, H, w) for p_seq in p_seqs, w in w_seq])
end

# abilities of the players(indexed in order)
H = [1,1,1]
#H = [47, 82, 65, 99, 2, 14, 9] # H is for 'habilidade'


function solve_problem(H)
  n = length(H) # number of players
  if n > 7
    return solve_problem_sameval(H)
  end
  MS = markov_states(n, H)
  m = length(MS)
  # S matrix m × m
  rows_S = Int64[]
  cols_S = Int64[]
  vals_S = Float64[]

  # T matrix m × n
  rows_T = Int64[]
  cols_T = Int64[]
  vals_T = Float64[]

  blksize = factorial(n)
    
  for i in 1:m
      s1 = MS[i]
      p1 = s1.player_sequence
      x = s1.ability_sequence[1] # master's ability
      y = s1.ability_sequence[2] # challenger's ability

      # Probability that the challenger will become the new master
      # Note: when this state transition occurs, the new state
      # has w = 0, i.e. the amount of times the master has won is reset to 0
      b = vcat(p1[2], p1[3:end], p1[1])
      j = nthperm(b)
      push!(rows_S, i)
      push!(cols_S, j)
      push!(vals_S, prob_lose(x, y)^2)

      
      #@assert (nthperm(p1) + blksize * s1.w) == i
      if s1.w < n - 2
          # Probability that the master will keep his title
          # Note: when this state transition occurs, the new state
          # has w += 1, i.e. the amount of times the master has won is increased by 1
          a = vcat(p1[1], p1[3:end], p1[2])
          j = nthperm(a) + (s1.w + 1) * blksize
          push!(rows_S, i)
          push!(cols_S, j)
          push!(vals_S, 1 - prob_lose(x, y)^2)
      elseif s1.w == n - 2
          # Probability that the master will transition to the final state(absorption)
          jT = p1[1]
          push!(rows_T, i)
          push!(cols_T, jT)
          push!(vals_T, 1 - prob_lose(x, y)^2)
      end
  end

  # S is a m × m matrix of 
  # state transition probabilities
  S = sparse(rows_S, cols_S, vals_S, m, m)

  # T: matrix which has the transition probabilities
  # to the final states
  T = sparse(rows_T, cols_T, vals_T, m, n);

  IS = Float64.(1.0I - S)

  soln = IS \ collect(T[:,1])

  soln[1]
end


# Same-ability problem: we can simplify the markov states
struct SameProbMarkovState{T<:Integer}
  # Stores the first Mario Kart Master's position
  # It can go from 1 to n, where pos=1 means it is the current Master
  pos::T
  # Amount of times the current Mario Kart Master has won
  w::T

  function SameProbMarkovState{T}(pos::T, w::T) where T <: Integer
      new(pos, w)
  end
end

SameProbMarkovState(pos, w::T) where T <: Integer = SameProbMarkovState{T}(pos, w)

# generates all possible markov states for a given n
function same_prob_markov_states(n)
  # all possible positions
  pos_seq = 1:n
  # all possible amount of times the current Mario Kart Master has won
  w_seq = 0:(n-2)

  vec([SameProbMarkovState(pos, w) for pos in pos_seq, w in w_seq])
end

function state_index(pos, w, n)
  pos + w * n
end

function solve_problem_sameval(H)
  n = length(H) # number of players
  MS = same_prob_markov_states(n)
  m = length(MS)
  # S matrix m × m
  rows_S = Int64[]
  cols_S = Int64[]
  vals_S = Float64[]

  # T matrix m × 2
  rows_T = Int64[]
  cols_T = Int64[]
  vals_T = Float64[]
    
  for i in 1:m
    s1 = MS[i]
    p1 = s1.pos
    w1 = s1.w

    p2 = -1
    w2 = 0
    if p1 == 1
      p2 = n
    elseif p1 == 2
      p2 = 1
    else
      p2 = p1 - 1
    end
    j = state_index(p2, w2, n)
    push!(rows_S, i)
    push!(cols_S, j)
    push!(vals_S, 1/4)

    if w1 < n - 2
        if p1 == 1
          p2 = 1
          w2 = w1 + 1
        elseif p1 == 2
          p2 = n
          w2 = w1 + 1
        else
          p2 = p1 - 1
          w2 = w1 + 1
        end
        j = state_index(p2, w2, n)
        push!(rows_S, i)
        push!(cols_S, j)
        push!(vals_S, 3/4)
    else
        j = if p1 == 1
          1
        else
          2
        end
        push!(rows_T, i)
        push!(cols_T, j)
        push!(vals_T, 3/4)
    end
  end

  # S is a m × m matrix of 
  # state transition probabilities
  S = sparse(rows_S, cols_S, vals_S, m, m)

  # T: matrix which has the transition probabilities
  # to the final states
  T = sparse(rows_T, cols_T, vals_T, m, 2);

  # Full transition matrix
  P = [S T; zeros(2,m) I]

  @show P
  
  # Iterative method
  # max_error = 1e-7
  # max_iter = 1000
  # x = zeros(m+2)
  # x[1] = 1.0
  # x = x'
  # for _ in 1:max_iter
  #   x_prev = x
  #   x = x * P
  #   if norm(x - x_prev) < max_error
  #     break
  #   end
  # end

  # x

  IS = Float64.(1.0I - S)

  soln = IS \ collect(T[:,1])

  soln[1]
end

using Printf

function read_file(file)
  lines = readlines(file)
  for l in lines
    H = parse.(Int, split(l))
    @printf "%.16f\n" solve_problem(H)
  end
end
