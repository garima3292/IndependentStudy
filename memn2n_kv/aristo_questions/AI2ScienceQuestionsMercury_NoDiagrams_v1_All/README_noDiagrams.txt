AI2 Science Questions Mercury - Without Diagrams v1

*****
* Do not distribute * 
The terms of this data set’s license agreement stipulate that this data should not be distributed except by the Allen Institute for Artificial Intelligence (AI2), and only after acceptance of the End User License Agreement. All parties interested in acquiring this data must request it from AI2 directly and accept the EULA (also included with this data set). Please visit allenai.org/data for more information.

Please contact ai2-data@allenai.org with any questions regarding this data set or to request updates about future data sets.
*****


About
——
This data set contains science questions provided under license by a research partner affiliated with AI2. These are English language questions that span several grade levels as indicated in the files. Each question is a 4-way multiple choice structure. 
These questions are text-only. They come pre-split into Train, Development, and Test sets. They come in two formats, CSV and JSON. The CSV files contain the full text of the question and its answer options in one cell. The JSON files contain a split version of the question, where the question text has been separated from the answer options programatically.
The question counts are as follows:

Exam01-Elementary-DMC-Train: 574
Exam01-Elementary-DMC-Dev: 143
Exam01-Elementary-DMC-Test: 717

Exam01-MiddleSchool-DMC-Train: 1583
Exam01-MiddleSchool-DMC-Dev: 485
Exam01-MiddleSchool-DMC-Test: 1639


Columns of the CSV
——
questionID: Unique identifier for the question.
originalQuestionID: Legacy ID used within AI2.
totalPossiblePoint: The point value of the question for grading purposes.
AnswerKey: The letter signifying the correct answer option for the question.
isMultipleChoice: 1 indicates the question is multiple choice.
includesDiagram: 0 indicates the question does not include a diagram.
examName: The name of the source exam for these questions.
schoolGrade: The intended grade level for the question.
year: The year the questions were sourced for AI2.
question: The question and its answer options. Each answer option is indicated by a letter in parentheses, e.g., (A) and (B).
subject: The question’s subject; this is left blank in this data set.
category: Whether the question is a Train, Dev, or Test question.


Structure of the JSON
——
The JSON files contain the same questions split into the “stem” of the question (the question text) and then the various answer “choices” and their corresponding labels (A, B, C, D). The questionID is also included.
