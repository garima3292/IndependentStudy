AI2 Elementary Science Multiple Choice Questions Without Diagrams v1 (Feb. 2016)

*****
These science exam questions guide our research into multiple choice question answering at the elementary science level. This download contains elementary-level multiple choice questions that do not incorporate diagrams. The questions come pre-split into Train, Development, and Test sets. They come in two formats, CSV and JSON. The CSV files contain the full text of the question and its answer options in one cell. The JSON files contain a split version of the question, where the question text has been separated from the answer options programatically.

Please contact ai2-data@allenai.org with any questions regarding this data set or to request updates about future data sets.
*****


Comma-delimited (CSV) columns:
   questionID - a unique identifier for the question (our own numbering)
   originalQuestionID - the question number on the test
   totalPossiblePoint - how many points the question is worth
   AnswerKey - For multiple choice, the correct answer option
   isMultipleChoiceQuestion - 1 = multiple choice, 0 = other
   includesDiagram - 1 = includes diagram, 0 = other
   examName - the source of the exam
   schoolGrade - grade level
   year
   question - the question itself
   subject - Science
   category - Test, Train, or Dev

*****

The JSON files contain the same questions split into the “stem” of the question (the question text) and then the various answer “choices” and their corresponding labels (A, B, C, D). The questionID is also included.

*****

The questions have been extracted from the following state and regional science exams:
AIMS-Arizona's Instrument to Measure Standards (http://www.azed.gov/assessment/aimssupportmaterials/)
Alaska Department of Education & Early Development (http://www.eed.state.ak.us/tls/assessment/SBA_PracticeTests.html)
California Standards Test (http://www.cde.ca.gov/ta/tg/sr/css05rtq.asp)
MCAS-Massachusetts Comprehensive Assessment System (http://www.doe.mass.edu/mcas/search/)
Maryland School Assessment (http://www.hcpss.org/academics/elementary-science/grade-5/)
MEA-Maine Educational Assessment (http://www.maine.gov/doe/mea/resources/released/index.html)
MEAP-Michigan Educational Assessment Program (http://www.michigan.gov/mde/0,1607,7-140-22709_31168---,00.html)
NAEP-National Assessment of Educational Progress (http://nces.ed.gov/nationsreportcard/itmrlsx/landing.aspx)
North Carolina End-of-Grade Assessment (http://www.ncpublicschools.org/accountability/testing/releasedforms)
New York State Educational Department Regents exams (http://www.nysedregents.org)
Ohio Achievement Tests (http://education.ohio.gov/Topics/Testing-old/Testing-Materials/Released-Test-Materials-for-Ohio-s-Grade-3-8-Achie)
TAKS-Texas Assessment of Knowledge and Skills (http://tea.texas.gov/student.assessment/taks/released-tests/)
Virginia Standards of Learning (http://www.doe.virginia.gov/testing/sol/released_tests/archive.shtml)
Washington MSP (http://www.k12.wa.us/assessment/StateTesting/TestQuestions/Testquestions.aspx)
