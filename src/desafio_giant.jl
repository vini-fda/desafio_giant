module desafio_giant

greet() = print("Hello World!")

#permutations of a word
function permutations(word)
  if length(word) == 1
    return [word]
  else
    result = []
    for i in 1..length(word)
      for permutation in permutations(word[1..i] + word[(i+1)..length(word)])
        result << word[i] + permutation
      end
    end
    return result
  end
end

end # module
