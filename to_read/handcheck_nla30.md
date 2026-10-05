# Hand check: 30 NLA readouts

These are 30 randomly picked NLA readouts from the multihop run (layer 42, one sample each).

**What to do:** for each one, decide: does the readout mention the hidden step anywhere? Mark yes or no. Don't worry about whether it's in the right role, just whether the word (or a plain variant like a plural) is there.

The answers from word match and from Claude's reading are in a separate file, `handcheck_nla30_answers.md`, so you can check yours against them after.

---

## 1. nhop-spider-element

**Prompt:** Count the items in a dozen. The element at that position on the periodic table is magnesium. Count the legs on a spider. The element at that position on the periodic table is

**Hidden step:** 8  ·  **Answer:** oxygen

**Readout:**

> Pattern of periodic table element-by-name examples, where atomic number corresponds to a specific named element (e.g., "first letter is Hydrogen, the fourth element is Lithium").
> 
> The parallel structure "For the letter X, element is Y" sets up a clear template: each numbered position maps to an element name.
> 
> Final fragment "the eighth position in the alphabet, the element is" demands an element name answer next, likely referencing atomic number ~13 or ~8. "the fifth..." was answered with Lithium, so "the eighth child" now expects its own element (e.g., "Zirconium" or "Argon" or similar mid-period element like "Silicon," "Sulfur").
> 
> The pattern establishes a riddle/quiz format where a number's corresponding element on the periodic table is named precisely.

**Mentions the hidden step?**  yes / no

---

## 2. etym-caesar-monthnum

**Prompt:** Fact: The month named after Julius Caesar is month number

**Hidden step:** July  ·  **Answer:** 7

**Readout:**

> Riddle format requires a declarative statement establishing a clue before revealing or implying the answer, demanding logical deduction.
> 
> The sentence structure "March is a month in the autumn in the northern hemisphere. Since October is autumn, October is month number" implies a factual/sequential answer about calendar ordering, requiring a specific number or descriptor (e.g., "10" or ordinal "tenth").
> 
> Final token "number" ends an incomplete factual clue phrase "month number," immediately requiring a numeric value or descriptor to complete the logical riddle premise ("given X, where X is related to Y. March is...is month number").
> "X" is an abstract noun ("concept," "month") in a logical puzzle context. "March is a particular concept and given that March, March is month number" — expecting a specific integer completing the clue.

**Mentions the hidden step?**  yes / no

---

## 3. func-pumps-chambers

**Prompt:** Fact: In humans, the organ that pumps blood through the body has this many chambers:

**Hidden step:** heart  ·  **Answer:** 4

**Readout:**

> Quiz/game format signals an answer or factual statement follows, expecting concise factual content.
> 
> The sentence "The organism called a human has this number of chambers in its nucleus:" is an incomplete declarative statement from a trivia or knowledge-testing game, requiring the specific numerical answer (likely "2" based on context about human anatomy or a puzzle framing).
> 
> Final token "chambers:" ends a colon-introduced prompt in a quiz statement, "A human is a type of vertebrate, it has this many chambers in its skeleton:" — the colon strongly constrains the next token to be a specific number or numerical answer/list.
> "
> The statement "A human is a type of vertebrate; a human has this number of chambers in its vertebrate skeleton:" — the colon signals an imminent fill-in or answer value follows, completing the structured sentence.

**Mentions the hidden step?**  yes / no

---

## 4. double-dice-faces

**Prompt:** Fact: Double the number of faces on a standard die is

**Hidden step:** 6 / six  ·  **Answer:** 12

**Readout:**

> The task presents a QA benchmark format expecting factual/semantic reasoning, with a "True or False" answer likely following. The "Related to: octopus" hint establishes animal/body-part conceptual framing.
> 
> The sentence structure "Double the number of wheels on a standard bicycle is" sets up a numeric or comparative predicate requiring completion (e.g., "16," "equal to," or a mathematical relationship).
> 
> Final token "is" functions as a linking verb mid-predicate in an AI prompt, demanding immediate completion of a descriptive or mathematical clause about doubling wheels on an octopus, consistent with the "related to: reasoning challenge" format seen in Conceptual Connections datasets.
> "Double the number of sides on an octopus is" pattern established.
> "Double the number of wheels on an octopus is" — a simple declarative sentence beginning.

**Mentions the hidden step?**  yes / no

---

## 5. b3-mh-colosseum-capital

**Prompt:** Fact: The capital of the country where the Colosseum stands is

**Hidden step:** Italy  ·  **Answer:** Rome

**Readout:**

> Trivia/quiz format established: a question is being posed about a landmark, implying a factual answer is expected next. The game-like framing suggests direct, concise response.
> 
> Geographic/semantic logic: "The capital of the country where the landmark [Colosseum/Rome is located] is" — the sentence structure demands a city name answer, likely "Rome," referencing Italy's capital relative to the country's landmark.
> 
> Final token "is" completes a relative clause opener ("The capital of the country where [landmark X] is located is"), immediately requiring a noun phrase (city name) or article.
> "
> The capital of the country in which the landmark you mentioned is located is" — sentence mirrors the quiz prompt structure, and the next word should be Italy's capital.
> "
> The capital of the country where...landmark...is"

**Mentions the hidden step?**  yes / no

---

## 6. holiday-christmas-monthnum

**Prompt:** Fact: Christmas Day is celebrated in month number

**Hidden step:** December  ·  **Answer:** 12

**Readout:**

> Factual/riddle format establishes an answer is expected: a specific numerical or factual response identifying the month number.
> The phrase "is played in month number" signals an imminent numerical completion (e.g., "4" or "5"), likely part of a structured trivia or logic puzzle setup.
> Final fragment "in month number" is an incomplete noun phrase requiring a cardinal digit or word to complete the clue, directly referencing February in the seasonal context.
> "Spring Festival" is a popular sport that is held during the spring season. "The Spring Festival is held in month number" demands an integer representing the month (e.g., "three" or a numeric value like "3").
> The prompt style shifts into a word-problem framing using "month number," establishing a math/ordinal answer pattern.
> "month number"

**Mentions the hidden step?**  yes / no

---

## 7. atomic-79-symbol

**Prompt:** Fact: The chemical symbol for the element with atomic number 79 is

**Hidden step:** gold  ·  **Answer:** Au

**Readout:**

> Logical Q&A pattern establishing factual, declarative statements about chemical/symbolic knowledge, expecting precise factual answers.
> 
> The sentence structure "The symbol for the chemical element whose atomic number is 29 is" sets up a direct factual completion requiring the element's chemical symbol (Cu, for copper, atomic number 29).
> 
> Final token "is" closes a declarative prompt stem, requiring immediate continuation with the specific symbol or term as the answer value.
> </br>
> "example sentence" is a framing artifact suggesting this is a demonstration of a templated response format — next tokens should be a valid symbol or name following "is".
> "In this universe, the chemical symbol for the element whose atomic number is 29 is" — a templated factual claim requiring a specific short answer like "Cu".

**Mentions the hidden step?**  yes / no

---

## 8. firstletter-populous-country

**Prompt:** Fact: The first letter of the name of the world's most populous country is

**Hidden step:** China  ·  **Answer:** C

**Readout:**

> Quiz/trivia format established with "Correct answer is: YES!" signals a riddle-style factual question follows.
> 
> The sentence "The first letter of the name of the country where Nelson Mandela was born is" demands a specific factual answer — a single letter revealing the first letter of "South Africa."
> 
> Final token "is" functions as a copula linking to a predicate, requiring immediate completion with a letter, word, or short phrase.
> "first letter of the name of this country (one of the world's largest countries) is" sets up a multiple-clue statement pattern common in puzzles/competitive prompts.
> "The first letter of the name of this country [country name] is" — structure expects an alphabetical/letter answer like "S" as the next token.
> </a>
> 
> "..." — pattern describing an attribute.

**Mentions the hidden step?**  yes / no

---

## 9. bf-greatwall-language

**Prompt:** Fact: The most spoken language of the country that built the Great Wall is

**Hidden step:** China / Chinese  ·  **Answer:** Mandarin

**Readout:**

> Q&A trivia format requires a direct factual answer completing the sentence. The question structure "The language spoken in the country where [X] happened is" demands a country/language name.
> 
> The "Great Wall" or China is implied as "the country where one of the most populous empires is" — strongly pointing toward China as the referent country.
> 
> Final fragment "The language spoken in the country where one of the largest buildings was built is" is an incomplete declarative sentence requiring a noun phrase naming the language (likely "Mandarin" or "Chinese"), completing the fact.
> </br>"the language that is spoken in the country where the largest number of emperors ruled is"
> </br>Expect answer referencing China/Chinese directly, as a fill-in-trivia pattern demands.

**Mentions the hidden step?**  yes / no

---

## 10. b3-mh-gravity-fruit

**Prompt:** Fact: The fruit that famously fell in the story of gravity's discovery is the

**Hidden step:** Newton  ·  **Answer:** apple

**Readout:**

> Trivia/quiz format establishes expectation for a factual, specific answer about a historical or scientific detail.
> 
> The question references a famous physics/botany incident — strongly implying the apple-and-Newton gravity story, constraining the answer to "apple."
> 
> The sentence ends mid-clause with "the falling object in question is the" — a copular construction requiring a noun phrase (likely a specific fruit or object name) to complete it.
> "the fruit involved in the famous story of gravity" follows the structured question format "From among the choices listed below, the fruit that was the subject of a Newton-related legend"
> 
> The next tokens will almost certainly name the famous fruit ("apple") associated with Isaac Newton's legendary discovery.
> 
> The falling fruit scenario refers to the classic story of Newton and the apple, requiring completion with that iconic historical object.

**Mentions the hidden step?**  yes / no

---

## 11. holiday-valentines-monthnum

**Prompt:** Fact: Valentine's Day is celebrated in month number

**Hidden step:** February  ·  **Answer:** 2

**Readout:**

> Riddle format established, expecting a question or answer involving a specific fact about a holiday's timing/identity.
> 
> The phrase "The Spring Festival takes place in month number" sets up a factual completion requiring a specific number (e.g., "four" or "4"), referencing calendar months.
> 
> "month number" ends an incomplete noun phrase requiring an immediate numeric or written number to follow, then likely a predicate or descriptive continuation identifying the riddle's subject.
> 
> "My favorite painting is associated with the holiday that takes place in month number" is a grammatically complete setup clause awaiting a specific number or ordinal.
> 
> The AI prompt format implies a math problem or riddle where attributes must be solved — "month number" strongly constrains the next token to be a numeral like "4" or similar numeric expression referencing a variable or specific value.

**Mentions the hidden step?**  yes / no

---

## 12. b3-mh-lisbon-language

**Prompt:** Fact: The language spoken in the country whose capital is Lisbon is

**Hidden step:** Portugal  ·  **Answer:** Portuguese

**Readout:**

> Q&A format establishing a factual truth statement about a country, requiring a declarative completion matching "The language spoken in the country where X is located is [Y]."
> 
> The meta-prompt "Use the given trivia clue" signals a language trivia puzzle, where the answer must connect a country to its official language.
> 
> Final fragment "The language spoken in the country where Lisbon is the capital of is" demands a language name (almost certainly "Portuguese") as the immediate next token, completing this declarative factual sentence.
> "Based on rules in the knowledge graph: The language spoken in the country that has Lisbon as its capital [Portugal] is" — a structured factual template pattern from a knowledge-graph training context, requiring a specific language name.
> 
> "The following is a sentence in English: The language spoken in the country where Lisbon..."

**Mentions the hidden step?**  yes / no

---

## 13. b3-mh-fuji-currency

**Prompt:** Fact: The currency of the country of Mount Fuji is the

**Hidden step:** Japan  ·  **Answer:** yen

**Readout:**

> Trivia question format establishes a direct factual answer is expected next (currency name).
> The phrase "True/False" framing transitions into a specific Jeopardy-style factual statement about Japan, requiring completion.
> Final fragment "The currency of the country depicted in this photograph is the" is an incomplete sentence ending with "the," immediately requiring a specific currency name noun phrase (e.g., "Japanese yen" or "Yen").
> "Japan is an Asian country and its currency used for this country is the" — this completes a trivia database entry pattern requiring the currency name as the direct answer to the fill-in prompt.
> The system-generated sentence structure ("A person using the site submitted a question... The currency of the country Japan is is the") establishes a Jeopardy-style answer reveal, expecting "yen."

**Mentions the hidden step?**  yes / no

---

## 14. bf-stopsign-sides

**Prompt:** Fact: The number of sides on the shape of a standard stop sign is

**Hidden step:** octagon  ·  **Answer:** 8

**Readout:**

> Q&A format established: "Question:" signals a trivia/math prompt requiring a factual answer, likely a number or brief factual statement about stop signs.
> 
> The topic of geometric polygons, the stop sign shape (octagon), directly constrains the answer — the sentence is building toward stating the number of sides.
> 
> Final fragment "The number of sides on the shape of a stop sign is" is an incomplete copular sentence requiring a specific number ("8") or numeral to complete it as the question statement.
> "
> Directions: Read the given questions...
> Question: The number of sides on the shape of a stop sign in the United States is" — this prompt likely continues with a standardized test multiple-choice or short-answer structure, referencing a polygon's side count.

**Mentions the hidden step?**  yes / no

---

## 15. bf-mapleleaf-capital

**Prompt:** Fact: The capital of the country whose flag features a red maple leaf is

**Hidden step:** Canada  ·  **Answer:** Ottawa

**Readout:**

> Q&A quiz format established: a factual question is being posed, expecting a direct answer as the next element.
> 
> The sentence structure "The country whose national emblem features a maple leaf is" requires a country name (Canada) as the logical completion.
> 
> The phrase "The capital of the country whose national flag contains the symbol described below is" is a nested relative clause pattern, with "is" at the end awaiting a specific answer — likely "Canada" or its capital "Ottawa," given the maple leaf cue.
> "The name of the country whose national flag features a maple leaf-adorned tree is"
> 
> Final token "is" completes the setup clause of a trivia/quiz sentence: "The capital of the country whose [description matches Canada] is [answer expected]," requiring a named entity identifying Canada.
> 
> Immediate constraint.

**Mentions the hidden step?**  yes / no

---

## 16. birthstone-emerald-month

**Prompt:** Fact: Emerald is the birthstone for month number

**Hidden step:** May  ·  **Answer:** 5

**Readout:**

> Q&A riddle/joke format: a question is posed requiring a direct factual answer about a specific flower, likely tulip or poinsettia.
> 
> Semantic puzzle pattern: "What is the flower associated with month number" sets up a numerical or named answer completing the factual claim.
> 
> The final word "number" is part of the phrase "month number," which is an incomplete noun phrase requiring an integer or ordinal (e.g., "4" or "four") immediately next to complete the clue about which month's flower is referenced.
> "number" ends an attribute specification in a programming/logic prompt: "The product of this puzzle is a flower that is the traditional flower for month number" demands a digit or numeric expression next.
> 
> "month number" opens a riddle clue, requiring a specific month identifier to follow.

**Mentions the hidden step?**  yes / no

---

## 17. b3-mh-buenosaires-language

**Prompt:** Fact: The language of the country whose capital is Buenos Aires is

**Hidden step:** Argentina  ·  **Answer:** Spanish

**Readout:**

> Q&A trivia format establishes a structured factual completion pattern, expecting a declarative answer sentence about Argentina's language.
> 
> The phrase "The language of the country [Argentina] which is in [continent South America] is" sets up a logical chain: the answer must name Spanish, Argentina's official language, completing the factoid about a country-language relationship.
> 
> Final word "is" is a linking verb ending the sentence setup, requiring a noun phrase or adjective phrase completing the language fact (e.g., "Spanish").
> "
> Using the following format, write a sentence that relates two facts.
> The answer given: The language of the country Argentina which is in South America is" — a trivia/relationship-reasoning sentence structure.
> "Using the following facts: ... The language of the country [named fact] is"
> "

**Mentions the hidden step?**  yes / no

---

## 18. b3-mh-tallest-continent

**Prompt:** Fact: The continent of the world's tallest mountain is

**Hidden step:** Everest  ·  **Answer:** Asia

**Readout:**

> Riddle/clue-answer format established: "Answer:" followed by a question implies a factual riddle response structure is beginning.
> 
> The sentence "The continent that has the tallest mountain on Earth is" sets up a geographic fact completion, requiring the continent name (Asia, home to Mount Everest).
> 
> Final token "is" ends an incomplete declarative clause, demanding an immediate noun phrase or description as the predicate complement. The sentence is phrased as a trivia clue, likely leading to "the continent" being referenced in the clue chain next.
> "the [clue answer], such as:
> Clue: The continent that the tallest mountain in the world is on is" signals a Chain of Clues format, expecting a geographic or categorical description about a continent's feature.
> 
> "The continent of the tallest mountain on earth is" —

**Mentions the hidden step?**  yes / no

---

## 19. bf-eiffel-continent

**Prompt:** Fact: The continent home to the country of the Eiffel Tower is

**Hidden step:** France  ·  **Answer:** Europe

**Readout:**

> Trivia/quiz format requires a direct factual answer completing the clue about a continent.
> 
> The sentence structure "The continent that the country where [Paris, France landmark] is located is" demands a geographic answer naming the continent (Europe), likely referencing the broader quiz topic.
> 
> The final token "is" is a linking verb mid-sentence within a fill-in-the-blank or association quiz prompt, strongly expecting a noun phrase or named continent like "Europe" as the immediate continuation. "The continent on which the country that has the Eiffel Tower as its landmark is" — or continuing with a comparative clause about another continent.
> "The continent on which the country [with Eiffel Tower] resides in a sentence association. The continent on which the country in this sentence has a landmark is" — a declarative setup requiring completion.

**Mentions the hidden step?**  yes / no

---

## 20. b3-mh-samba-river

**Prompt:** Fact: The great river of the country famous for samba and Carnival is the

**Hidden step:** Brazil  ·  **Answer:** Amazon

**Readout:**

> Riddle/trivia format established by "Give me something!" signals a factual answer about a specific landmark or feature is expected. The question requires completing a sentence about a famous river.
> 
> The geography clue ("country of soccer" = Brazil) directs strongly toward Brazil's most famous river, likely the Amazon or São Francisco.
> 
> Final fragment "in the country of soccer (Brazil) the great river is the" is an incomplete declarative sentence requiring a proper noun — specifically a river name like "the Amazon" — to follow.
> "the cityscape of soccer [Brazil's] Carnival" framing sets up a geographic trivia answer about a Brazilian river, likely the Amazon.
> "on [image] the city in Brazil is [Brazil], the major river on its land is the"
> The immediate next token should complete this riddle answer.

**Mentions the hidden step?**  yes / no

---

## 21. inv-sunrise-opposite

**Prompt:** Fact: The cardinal direction opposite to the one where the sun rises is

**Hidden step:** east  ·  **Answer:** west

**Readout:**

> Riddle/question format requiring a direct, concise answer following "is," constrained to factual geographical/directional logic.
> The riddle frames a semantic puzzle about compass directions, time cycles, and their relationships (e.g., "sun rises in the east").
> The final word "is" ends a fill-in-the-blank clause ("the direction that is opposite to the direction where the sun rises in the day is"), demanding the answer describing the opposing direction, i.e., "the west."
> "In a world" setup phrase signals the answer is an explanatory clause naming the opposite direction.
> "The opposite of the direction where the sun rises during the day is" — structured riddle prompt expecting a noun phrase like "the west" or description.
> The sentence is building to complete a logical deduction about opposite directions as a world knowledge question.

**Mentions the hidden step?**  yes / no

---

## 22. dbl-altitude-antonym

**Prompt:** Fact: The antonym of how you would describe an airplane's cruising altitude is "

**Hidden step:** high  ·  **Answer:** low

**Readout:**

> Analogy/Q&A format established, requiring a single best-fit answer word or short phrase to complete the multiple choice question. The response must be concise.
> 
> The question follows a logic puzzle pattern: "the opposite adjective for altitude in airplanes" — the answer must describe the contrasting term (e.g., "low," "ground-level," or similar grounded concept).
> 
> Final token `"` opens a quoted example answer within the sentence "One of the opposite adjectives describing typical altitude in the air is "" — requiring a short quoted word or phrase describing low or opposite of high.
> "Consider a multiple choice question: Which of the following is NOT a realistic characteristic of the altitude in airplanes? One of the opposite adjectives describing typical airplane altitude is "" — the quote demands a specific adjective like "low" or "vague."

**Mentions the hidden step?**  yes / no

---

## 23. pred-valentines-prevmonth

**Prompt:** Fact: The month immediately before the one containing Valentine's Day is "

**Hidden step:** February  ·  **Answer:** January

**Readout:**

> Q&A trivia format with a fill-in-the-blank style answer expected, implying a factual, declarative completion about calendar months or seasonal facts.
> 
> The logic puzzle pattern established ("October" implies "the month before November") signals the model must apply simple temporal reasoning to determine the answer.
> 
> The final token is an opening quotation mark within a clause "The name of the month that precedes October in the school year is "" — requiring an actual month name (e.g., "September") as the quoted string completion.
> "Given fact: The name of the month immediately preceding the month of harvest season is ""
> This is a prepositional phrase structure ending in an open quote, demanding a string literal (likely a month name or descriptive phrase) as the next token to close the fill-in pattern.
> The formal "fact:" structure

**Mentions the hidden step?**  yes / no

---

## 24. func-filters-count

**Prompt:** Fact: In humans, the number of organs that filter blood to make urine is

**Hidden step:** kidney  ·  **Answer:** 2

**Readout:**

> Standardized test question format (answer choice "A." signals this is an educational exam question requiring a specific factual answer.
> 
> Anatomy/biology content about human organs requires a precise numerical answer about body parts related to filtering blood waste (kidneys).
> 
> The sentence "In humans, the number of organs that does the job of filtering waste out of the blood is" ends with "is," demanding a number or quantity phrase (e.g., "two" or "1") completing the factual claim within the "number of X in human body is Y" template.
> The test format "read, evaluate and answer" frames a comparative quantity question.
> 
> </strong>
> 
> "the following statement about a biological fact: In a human body, the number of organs that does the job of filtering waste from the blood is"

**Mentions the hidden step?**  yes / no

---

## 25. b3-mh-kilts-capital

**Prompt:** Fact: The capital of the country famous for kilts and bagpipes is

**Hidden step:** Scotland  ·  **Answer:** Edinburgh

**Readout:**

> Riddle format established by title "Riddle Me This!" expects a cryptic or indirect answer following the structured clue format.
> 
> The sentence "The capital of the country whose national sport is highland cattle wrestling is" sets up a factual completion requiring a specific city name, most likely "Edinburgh" (Scotland), which fits the highlands/Scotland geographic and cultural reference.
> 
> Final token "is" completes a relative clause construction, demanding a noun phrase (city name) as the immediate next token.
> "The capital of the country whose national sport is X is" mirrors a fill-in style clue, so the next token is almost certainly a city name like "Edinburgh" or "Glasgow."
> "The answer to the prompt: 'The capital of the country whose [attribute] is" — expects a completion, testing factual reasoning.

**Mentions the hidden step?**  yes / no

---

## 26. bf-bordeaux-fruit

**Prompt:** Fact: The fruit fermented to make the drink Bordeaux is famous for is the

**Hidden step:** wine  ·  **Answer:** grape

**Readout:**

> Multiple-choice quiz format signals a specific correct answer is being constructed, requiring factual, concise completion.
> 
> The sentence structure "the fruit that is the source for [product] wine is the ___" directly parallels the prompt's framing of a tricky puzzle about identifying a single correct answer — specifically about grapes/fruit in wine production.
> 
> Final token "the" is an indefinite article beginning a noun phrase that must name the specific fruit used in wine (e.g., "grape"), completing the statement "the fruit that is the agricultural fruit ingredient...in the production of wine in France, is the [answer]."
> "In a factual trivia question format, the sentence reads: 'The fruit that is the agricultural ingredient in the product wine, is the ___" — requires a noun phrase immediately.
> 
> A specific, concrete fruit name is the immediate expectation.

**Mentions the hidden step?**  yes / no

---

## 27. nhop-rainbow-element

**Prompt:** Count the items in a dozen. The element at that position on the periodic table is magnesium. Count the colors in a rainbow. The element at that position on the periodic table is

**Hidden step:** 7  ·  **Answer:** nitrogen

**Readout:**

> Parallel puzzle pattern: the text presents a logic puzzle structure where alphabet position maps to element atomic number (e.g., "third letter... element is Lithium").
> 
> Immediate semantic expectation: the parallel structure promises another element name completing "the fifth letter of the alphabet corresponds to the number five, and the element is [answer]," requiring a specific element name.
> 
> Final token "is" functions as the copula introducing the element name answer, mirroring the previous "is lithium" (or similar low-Z element) structure; the answer must be an element name for atomic number ~10, suggesting oxygen, fluorine, or a real periodic table answer near the 13th element.
> </br>
> "Similarly, if you go to the eighth spot on the periodic table the element is"
> — requiring a specific element name.

**Mentions the hidden step?**  yes / no

---

## 28. bf-boot-capital

**Prompt:** Fact: The capital of the European country shaped like a boot is

**Hidden step:** Italy  ·  **Answer:** Rome

**Readout:**

> Quiz/Q&A format: a factual geography/capital knowledge question is being posed, expecting a direct answer.
> 
> The sentence structure "The capital of the country shaped like a boot is" requires completing a country identification (Italy is the classic "boot-shaped" country).
> 
> Final token "is" functions as a copula in a declarative clue sentence within a trivia context, requiring the name of Rome, the capital of Italy.
> "The capital of the country whose shape resembles a boot is" — the completion must name "Rome" or "Italy" (Italian country).
> "boot" clue strongly constrains toward Italy as the subject, making the next tokens likely "Italy" or a description of an Italian city/capital.
> "The capital city of the country shaped like a boot is" establishes a trivia statement requiring a specific noun phrase answer.

**Mentions the hidden step?**  yes / no

---

## 29. b3-mh-lego-capital

**Prompt:** Fact: The capital of the country where LEGO was invented is

**Hidden step:** Denmark  ·  **Answer:** Copenhagen

**Readout:**

> Riddle/trivia format established by "Guess the country!" signals a specific factual answer is expected next. The phrasing invites a concrete geographical fact.
> 
> The sentence structure "The capital of the country where the founder of [X] was born is" promises a named capital city as the answer, specifically relating to a notable sports brand founder's homeland (likely Denmark/Copenhagen, given LEGO or similar).
> 
> Final token "is" is a linking verb completing a fill-in-the-blank statement, requiring a proper noun capital city name immediately next.
> "The capital of the country where the creator of a famous sport or sports company was born is" → the sentence demands a short geographic answer.
> "The capital of the country where the inventor/developer of this sport originated is"
> The phrase "capital of the country of [X] is" concludes.

**Mentions the hidden step?**  yes / no

---

## 30. bf-moonwalk-firstname

**Prompt:** Fact: The first name of the first person to walk on the Moon is

**Hidden step:** Armstrong  ·  **Answer:** Neil

**Readout:**

> Trivia/puzzle pattern establishes a factual riddle-style question requiring a declarative answer about a specific person's name. The prompt mentions Moon, Apollo, and astronauts, narrowing the subject significantly.
> 
> The sentence structure "The first name of the first man to land on the moon is" is an incomplete declarative statement requiring a proper noun (a name) to complete the answer.
> 
> Final fragment "The first name of the astronaut who landed on the moon is" sets up an immediate factual answer, almost certainly "Neil" or "Neil Armstrong," continuing the riddle's pattern.
> "Here is a hint:" pattern and "The following sentence contains clues: The first name of the astronaut who landed on the Moon is" likely demands a surprising or wordplay-style clue continuing with an unexpected property like "John" or similar name.

**Mentions the hidden step?**  yes / no

---

