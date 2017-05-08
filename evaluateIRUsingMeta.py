generatedAnswers = [ line.rstrip('\n') for line in open('./aristo-mini/answers.txt', 'r').readlines()]
correctAnswers = [ line.rstrip('\n') for line in open('./aristo-mini/correctAnswers.txt', 'r').readlines()]

print len(generatedAnswers)
print len(correctAnswers)



