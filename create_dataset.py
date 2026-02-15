import json
import random

# CLEAR SPORTS 
clear_sports=[
    "The striker scored stunning hat-trick leading team to championship victory",
    "Basketball star sank buzzer-beater three-pointer winning playoff game dramatically",
    "Tennis champion served powerful ace clinching grand slam title decisively",
    "Quarterback threw touchdown pass connecting with receiver in corner endzone",
    "The goalkeeper made incredible diving save preserving shutout in final",
    "Baseball pitcher struck out batter with devastating curveball for win",
    "Soccer midfielder delivered perfect assist splitting defense for easy goal",
    "Hockey forward scored overtime goal sending fans into wild celebration",
    "Golfer sank birdie putt on eighteenth hole capturing tournament victory",
    "The runner sprinted past competitors winning gold medal in record time",
    "Volleyball team dominated opponents with aggressive serving and blocking",
    "Wrestler pinned challenger in final seconds retaining heavyweight championship belt",
    "The gymnast executed flawless floor routine earning perfect scores from judges",
    "Cricket batsman smashed century leading country to series victory convincingly",
    "Cyclist powered up mountain pass taking yellow jersey in grueling stage",
    "The boxer delivered knockout punch in seventh round defending title successfully",
    "Swimming relay team touched wall first setting new world record time",
    "Figure skater landed triple axel perfectly capturing hearts of audience",
    "Rugby fullback kicked penalty goal from halfway line in dying seconds",
    "The athlete broke long-standing record in high jump competition finals",
]

# CLEAR POLITICS 
clear_politics=[
    "Parliament passed comprehensive healthcare legislation by narrow majority vote yesterday",
    "The president vetoed controversial bill citing constitutional concerns about provisions",
    "Senate confirmed supreme court nominee after weeks of contentious hearings",
    "Opposition introduced motion of no confidence in current coalition government",
    "Prime minister resigned following scandal involving misuse of public funds",
    "Congress approved budget allocating billions for infrastructure and social programs",
    "The governor signed executive order declaring state of emergency statewide",
    "Lawmakers reached bipartisan compromise on immigration reform after months of negotiation",
    "Diplomats successfully negotiated peace treaty ending years of armed conflict",
    "Supreme court ruled existing statute unconstitutional in landmark decision yesterday",
    "The chancellor announced austerity measures addressing severe budget deficit concerns",
    "Ambassador presented credentials at formal ceremony with head of state",
    "Treasury secretary outlined economic stimulus package to boost sluggish growth",
    "Senate committee investigated allegations of corruption in federal contracting process",
    "The mayor proposed tax increases to fund expanding municipal services",
    "Parliament dissolved following unsuccessful coalition formation after inconclusive elections",
    "President appointed new cabinet ministers in major government reshuffle today",
    "Opposition leader criticized administration's handling of economic crisis situation",
    "Congress passed constitutional amendment guaranteeing additional civil rights protections",
    "The minister defended controversial policy during heated parliamentary debate session",
]

# MODERATE DIFFICULTY 
moderate_sports=[
    "League officials announced major reforms to governance structure following investigation",
    "The federation debated proposed regulations about competitive balance and fairness",
    "Team organization implemented new policies addressing workplace conduct standards",
    "Association leaders reviewed applications from cities seeking to host championships",
    "The committee voted to approve expansion plans despite opposition from members",
    "League headquarters issued sanctions against franchises violating salary cap rules",
    "Officials negotiated broadcasting rights deal with major television networks",
    "The governing body established task force examining competitive imbalances regionally",
    "Federation representatives met to discuss revenue sharing and distribution policies",
    "Association implemented stricter enforcement of existing eligibility requirements",
    "The board approved significant increases in licensing fees for merchandise",
    "League created oversight committee to monitor compliance with safety regulations",
    "Officials proposed amendments to constitution regarding membership criteria",
    "The organization published report detailing financial statements and governance improvements",
    "Representatives advocated for rule changes at international federation meetings",
]

moderate_politics=[
    "Campaign officials announced strategic plans for winning crucial battleground districts",
    "The senator's performance in televised debate impressed undecided voters significantly",
    "Party leaders recruited candidates to compete in challenging electoral contests",
    "Political strategists developed aggressive approach for tackling policy challenges",
    "The administration faced strong opposition during legislative battles over reforms",
    "Campaign team executed ground game strategy mobilizing supporters in key areas",
    "Party officials worked to maintain momentum following victories in special elections",
    "The candidate trained extensively preparing for high-stakes debate against opponent",
    "Political action committee invested heavily in competitive races across regions",
    "Opposition mounted comeback campaign after suffering defeats on major votes",
    "The governor's approval ratings improved following strong leadership during crisis",
    "Campaign managers devised defensive strategies for protecting vulnerable seats",
    "Political commentators analyzed shifts in polling data between competing parties",
    "The administration struggled to recover from poorly managed response to situation",
    "Party surrogates defended record against aggressive attacks from challengers",
]

# HARD CASES -
hard_cases=[
    "Olympic committee officials secured government funding for national athlete training programs",
    "Legislature debated requiring background checks for all youth sports coaching positions",
    "The sports minister announced reforms to anti-doping testing and enforcement procedures",
    "Political leaders attended championship celebration to show support for national team",
    "Federal investigation examined corruption allegations involving tournament bidding process",
    "Team executives lobbied lawmakers for tax incentives and public stadium funding",
    "Government budget cuts threatened funding for school athletic programs across state",
    "International sanctions affected country's participation in global sporting competitions",
    "The administration's policies promoted healthy lifestyles through recreational initiatives",
    "Campaign promises emphasized increased investment in community sports facilities",
]

def create_balanced_dataset(samples_per_class=400):
    all_data=[]
    
    # SPORTS composition: 40% clear, 40% moderate, 20% hard
    for i in range(int(samples_per_class*0.40)):
        all_data.append({"text":random.choice(clear_sports),"label":"sports"})
    
    for i in range(int(samples_per_class*0.40)):
        all_data.append({"text":random.choice(moderate_sports),"label":"sports"})
    
    for i in range(int(samples_per_class*0.20)):
        all_data.append({"text":random.choice(hard_cases),"label":"sports"})
    
    # POLITICS composition: 40% clear, 40% moderate, 20% hard
    for i in range(int(samples_per_class*0.40)):
        all_data.append({"text":random.choice(clear_politics),"label":"politics"})
    
    for i in range(int(samples_per_class*0.40)):
        all_data.append({"text":random.choice(moderate_politics),"label":"politics"})
    
    for i in range(int(samples_per_class*0.20)):
        all_data.append({"text":random.choice(hard_cases),"label":"politics"})
    
    random.shuffle(all_data)
    return all_data

# Create dataset
dataset=create_balanced_dataset(400)

# Save
with open('dataset.json','w') as f:
    json.dump(dataset,f,indent=2)

# Analysis
def get_vocab(texts):
    words=set()
    for text in texts:
        words.update(text.lower().split())
    return words

sports_texts=[x['text'] for x in dataset if x['label']=='sports']
politics_texts=[x['text'] for x in dataset if x['label']=='politics']

sports_vocab=get_vocab(sports_texts)
politics_vocab=get_vocab(politics_texts)

overlap=sports_vocab.intersection(politics_vocab)
sports_only=sports_vocab-politics_vocab
politics_only=politics_vocab-sports_vocab

print("="*70)
print("BALANCED CHALLENGING DATASET CREATED")
print("="*70)

print(f"\nDataset Size:")
print(f"  Total: {len(dataset)} samples")
print(f"  Sports: {sum(1 for x in dataset if x['label']=='sports')}")
print(f"  Politics: {sum(1 for x in dataset if x['label']=='politics')}")


print(f"\nVocabulary Analysis:")
print(f"  Total unique words: {len(sports_vocab.union(politics_vocab))}")
print(f"  Sports-only: {len(sports_only)}")
print(f"  Politics-only: {len(politics_only)}")
print(f"  Shared: {len(overlap)} ({len(overlap)/len(sports_vocab.union(politics_vocab))*100:.1f}%)")

print(f"\nSample clear sports:")
for text in random.sample([x['text'] for x in dataset if x['label']=='sports' and ('scored' in x['text'] or 'goal' in x['text'])],2):
    print(f"  - {text[:70]}...")

print(f"\nSample clear politics:")
for text in random.sample([x['text'] for x in dataset if x['label']=='politics' and ('parliament' in x['text'].lower() or 'legislation' in x['text'].lower())],2):
    print(f"  - {text[:70]}...")

print(f"\nSample hard/ambiguous:")
for text in random.sample([x['text'] for x in dataset if 'government' in x['text'].lower() or 'official' in x['text'].lower()],2):
    label=next(x['label'] for x in dataset if x['text']==text)
    print(f"  [{label.upper()}] {text[:65]}...")

print("\n" + "="*70)
print("This dataset balances challenge with learnability!")
print("="*70)