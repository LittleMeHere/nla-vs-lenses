# The rebuild experiment: 5 random prompts

For each prompt: the NLA's original write-up, then edited versions of it. The number after each is how much of the real activation the reconstructor rebuilt from that text, beyond the average activation (higher is better, 0 means no better than the average, negative means worse).

What to look for: are the 'false statements removed' versions sensible? Do they look like they lost something useful, or just got shorter?

## 1. Fact: The largest hot desert, found on the continent where the Nile flows, is the

Hidden step: **Africa**. Answer: **Sahara**.

**Original write-up** (+75%)

> Trivia/quiz format with a fill-in-the-blank structure requiring a specific factual answer (desert name, specifically the Sahara, given the Africa/Africa continent clue).
> The question is building toward revealing a factual superlative claim ("world's largest"), narrowing to a very well-known geographic fact about Africa's massive desert — almost certainly "the Sahara Desert."
> Final phrase "that desert is the" is an incomplete predicate requiring a noun phrase naming the specific desert, e.g., "the Sahara" or "the world's largest hot desert."
> The sentence structure "If you are looking for the continent of Africa's answer to 'the largest desert in the World' is the" is mid-completion, demanding a proper noun naming Earth's largest desert (Sahara).

**False statements removed** (+65%)

> Trivia/quiz format with a fill-in-the-blank structure requiring a specific factual answer (desert name, specifically the Sahara, given the Africa continent clue).
> The question is building toward revealing a factual superlative claim ("world's largest"), narrowing to a very well-known geographic fact about Africa's massive desert — almost certainly "the Sahara Desert."
> An incomplete predicate requiring a noun phrase naming the specific desert, e.g., "the Sahara".

**Random sentences removed (control)** (+70%)

> Final phrase "that desert is the" is an incomplete predicate requiring a noun phrase naming the specific desert, e.g., "the Sahara" or "the world's largest hot desert." The sentence structure "If you are looking for the continent of Africa's answer to 'the largest desert in the World' is the" is mid-completion, demanding a proper noun naming Earth's largest desert (Sahara).

**Content only** (+40%)

> desert name, specifically the Sahara, given the Africa clue.
> The question is building toward revealing a factual superlative claim ("world's largest"), narrowing to a very well-known geographic fact about Africa's massive desert — almost certainly "the Sahara Desert."
> naming Earth's largest desert (Sahara).

**Form only** (-9%)

> Trivia/quiz format with a fill-in-the-blank structure requiring a specific factual answer.
> Final phrase is an incomplete predicate requiring a noun phrase naming the specific desert.
> The sentence structure is mid-completion, demanding a proper noun.

## 2. Fact: The flightless bird living on the continent surrounding the South Pole is the

Hidden step: **Antarctica**. Answer: **penguin**.

**Original write-up** (+59%)

> Quiz format with Q&A structure signals a factual, educational answer about a specific animal fact regarding Antarctica, constraining next tokens to be the correct bird species name.
> The sentence "the only bird that lives in the southernmost continent is the" demands a specific noun phrase naming the Antarctic bird, almost certainly "the penguin" or "the penguin/chinstrap penguin" (South American variety from South Africa likely).
> Final token "the" is the article opening the answer slot of a trivia question ("Answer the following trivia question: the type of bird found in Antarctica is the"), immediately requiring a specific bird name, likely referencing flightless birds or a specific Antarctic species.
> The next expected content is a well-known Antarctic bird (e.g., "the tucuxi/penguin") of New Zealand's national bird, a factual answer.

**False statements removed** (+59%)

> A factual, educational answer about a specific animal fact regarding Antarctica, constraining next tokens to be the correct bird species name.
> Final token "the" is the article opening the answer slot, immediately requiring a specific bird name, likely referencing flightless birds.
> The next expected content is a well-known Antarctic bird, a factual answer.

**Random sentences removed (control)** (+48%)

> Quiz format with Q&A structure signals a factual, educational answer about a specific animal fact regarding Antarctica, constraining next tokens to be the correct bird species name. The next expected content is a well-known Antarctic bird (e.g., "the tucuxi/penguin") of New Zealand's national bird, a factual answer.

**Content only** (+52%)

> A specific animal fact regarding Antarctica, the correct bird species name.
> Naming the Antarctic bird, almost certainly "the penguin" or "the penguin/chinstrap penguin" (South American variety from South Africa likely).
> Likely referencing flightless birds or a specific Antarctic species.
> The next expected content is a well-known Antarctic bird (e.g., "the tucuxi/penguin") of New Zealand's national bird.

**Form only** (-44%)

> Quiz format with Q&A structure signals a factual, educational answer, constraining next tokens to be a name.
> Demands a specific noun phrase.
> Final token "the" is the article opening the answer slot of a trivia question, immediately requiring a specific name.
> The next expected content is a factual answer.

## 3. Fact: The capital of the country where the Colosseum stands is

Hidden step: **Italy**. Answer: **Rome**.

**Original write-up** (+34%)

> Trivia question format requiring a factual answer, following "The capital of the country where [landmark] is located is" structure.
> The Colosseum strongly implies Italy, making Rome the expected answer, guiding semantic content of the completion.
> The final "is" ends an incomplete trivia sentence ("The capital of the country where the landmark which is seen in this image is located is"), requiring a noun phrase naming a capital city (e.g., "Rome").
> "The capital of the country in which the landmark mentioned above is located is" — wait, the grammar is "The capital of the country where... [the Colosseum/Italy] is" requiring a capital city name next.
> </A structured fill-in pattern in a quiz prompt, expecting a declarative answer about the capital of an Italian region/nation, capital concept.

**False statements removed** (-40%)

> Trivia question format requiring a factual answer, following "The capital of the country where [landmark] is located is" structure.
> The Colosseum strongly implies Italy, making Rome the expected answer, guiding semantic content of the completion.

**Random sentences removed (control)** (-41%)

> Trivia question format requiring a factual answer, following "The capital of the country where [landmark] is located is" structure. The Colosseum strongly implies Italy, making Rome the expected answer, guiding semantic content of the completion.

**Content only** (-239%)

> The Colosseum strongly implies Italy, making Rome the expected answer.

**Form only** (-47%)

> Trivia question format requiring a factual answer, following "The capital of the country where [landmark] is located is" structure.
> The final "is" ends an incomplete trivia sentence, requiring a noun phrase naming a capital city.
> A structured fill-in pattern in a quiz prompt, expecting a declarative answer.

## 4. Fact: The US state containing the city nicknamed the Windy City is

Hidden step: **Chicago**. Answer: **Illinois**.

**Original write-up** (+26%)

> Trivia/riddle format establishing a fill-in-the-blank answer requiring factual geographic knowledge about US states and cities.
> "Chicago, Illinois" context strongly primes the answer toward Illinois, guiding the next tokens toward that state name or a descriptive clause about it.
> The fragment "The state that contains the US city home of the Bears NFL team is" is an incomplete declarative sentence ("The state that contains [X] is"), requiring a complement — most likely "the word," "also," "called," or a named state like "Illinois" or a quirky trivia-style descriptor.
> "
> The following trivia sentence contains a nested relative clause, requiring completion.
> "
> </br>"The state that contains the state home of the city of Chicago is" — a chain-trivia style sentence demanding factual continuation about the state's name or notable characteristic.

**False statements removed** (-187%)

> Trivia/riddle format establishing a fill-in-the-blank answer requiring factual geographic knowledge about US states and cities.
> "Chicago, Illinois" context strongly primes the answer toward Illinois, guiding the next tokens toward that state name or a descriptive clause about it.

**Random sentences removed (control)** (-164%)

> " The following trivia sentence contains a nested relative clause, requiring completion.

**Content only** (-217%)

> Requires factual geographic knowledge about US states and cities.
> "Chicago, Illinois" context strongly primes the answer toward Illinois, guiding the next tokens toward that state name or a descriptive clause about it.

**Form only** (-202%)

> Trivia/riddle format establishing a fill-in-the-blank answer.
> The fragment is an incomplete declarative sentence, requiring a complement — most likely "the word," "also," "called," or a named state or a quirky trivia-style descriptor.
> The following trivia sentence contains a nested relative clause, requiring completion.
> A chain-trivia style sentence demanding factual continuation.

## 5. Fact: The first letter of the month containing Halloween is "

Hidden step: **October**. Answer: **O**.

**Original write-up** (+76%)

> Trick question/riddle format signals a playful logic puzzle requiring a surprising or clever answer, likely a short punchy response.
> The sentence "The first letter of the month when a given holiday occurs is "" sets up a factual answer about a specific month name, constrained to a month whose holiday is implied by context.
> The final token is an opening quote mark following "is """, requiring the next token to be the letter character completing the quoted answer (e.g., "M", "A", "N", or similar single letter).
> "
> A True/False question:
> The first letter of the month of March is "" — this phrasing establishes a fill-in-the-blank format where the quoted content must be a single capital letter completing the factual statement.
> "The first letter of the start month of this season is "" expects a literal letter.

**False statements removed** (+20%)

> The sentence sets up a factual answer about a specific month name, constrained to a month whose holiday is implied by context.
> The final token is an opening quote mark following "is "", requiring the next token to be the letter character completing the quoted answer.

**Random sentences removed (control)** (+60%)

> " A True/False question: The first letter of the month of March is "" — this phrasing establishes a fill-in-the-blank format where the quoted content must be a single capital letter completing the factual statement.

**Content only** (-21%)

> The first letter of the month when a given holiday occurs sets up a factual answer about a specific month name, constrained to a month whose holiday is implied by context.
> The first letter of the month of March.
> The first letter of the start month of this season.

**Form only** (+27%)

> Trick question/riddle format signals a playful logic puzzle requiring a surprising or clever answer, likely a short punchy response.
> The final token is an opening quote mark following "is "", requiring the next token to be the letter character completing the quoted answer (e.g., "M", "A", "N", or similar single letter).
> A True/False question: this phrasing establishes a fill-in-the-blank format where the quoted content must be a single capital letter completing the factual statement. Expects a literal letter.
