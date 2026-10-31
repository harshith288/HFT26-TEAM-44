"""Build src/species_list.csv: the broad list used for BioCLIP and danger groups.

Takes every species-level mammal, bird and reptile in SpeciesNet's own labels, then adds
Indian wildlife, snakes and livestock that SpeciesNet lacks.

Usage: python src/scripts/build_species_list.py
"""
import csv
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# common name, class, order, family, genus, species
EXTRA = """
bengal tiger,mammalia,carnivora,felidae,panthera,tigris
asiatic lion,mammalia,carnivora,felidae,panthera,leo
indian leopard,mammalia,carnivora,felidae,panthera,pardus
clouded leopard,mammalia,carnivora,felidae,neofelis,nebulosa
jungle cat,mammalia,carnivora,felidae,felis,chaus
fishing cat,mammalia,carnivora,felidae,prionailurus,viverrinus
rusty-spotted cat,mammalia,carnivora,felidae,prionailurus,rubiginosus
indian wolf,mammalia,carnivora,canidae,canis,lupus
golden jackal,mammalia,carnivora,canidae,canis,aureus
bengal fox,mammalia,carnivora,canidae,vulpes,bengalensis
striped hyena,mammalia,carnivora,hyaenidae,hyaena,hyaena
asian black bear,mammalia,carnivora,ursidae,ursus,thibetanus
sun bear,mammalia,carnivora,ursidae,helarctos,malayanus
himalayan brown bear,mammalia,carnivora,ursidae,ursus,arctos
indian gray mongoose,mammalia,carnivora,herpestidae,urva,edwardsii
ruddy mongoose,mammalia,carnivora,herpestidae,urva,smithii
small indian civet,mammalia,carnivora,viverridae,viverricula,indica
common palm civet,mammalia,carnivora,viverridae,paradoxurus,hermaphroditus
honey badger,mammalia,carnivora,mustelidae,mellivora,capensis
smooth-coated otter,mammalia,carnivora,mustelidae,lutrogale,perspicillata
yellow-throated marten,mammalia,carnivora,mustelidae,martes,flavigula
indian elephant,mammalia,proboscidea,elephantidae,elephas,maximus
indian rhinoceros,mammalia,perissodactyla,rhinocerotidae,rhinoceros,unicornis
asiatic wild ass,mammalia,perissodactyla,equidae,equus,hemionus
blackbuck,mammalia,artiodactyla,bovidae,antilope,cervicapra
chinkara,mammalia,artiodactyla,bovidae,gazella,bennettii
chousingha,mammalia,artiodactyla,bovidae,tetracerus,quadricornis
indian muntjac,mammalia,artiodactyla,cervidae,muntiacus,muntjak
hog deer,mammalia,artiodactyla,cervidae,axis,porcinus
barasingha,mammalia,artiodactyla,cervidae,rucervus,duvaucelii
sangai,mammalia,artiodactyla,cervidae,rucervus,eldii
himalayan musk deer,mammalia,artiodactyla,moschidae,moschus,leucogaster
wild water buffalo,mammalia,artiodactyla,bovidae,bubalus,arnee
himalayan tahr,mammalia,artiodactyla,bovidae,hemitragus,jemlahicus
nilgiri tahr,mammalia,artiodactyla,bovidae,nilgiritragus,hylocrius
markhor,mammalia,artiodactyla,bovidae,capra,falconeri
bharal,mammalia,artiodactyla,bovidae,pseudois,nayaur
indian chevrotain,mammalia,artiodactyla,tragulidae,moschiola,indica
rhesus macaque,mammalia,primates,cercopithecidae,macaca,mulatta
gray langur,mammalia,primates,cercopithecidae,semnopithecus,entellus
lion-tailed macaque,mammalia,primates,cercopithecidae,macaca,silenus
nilgiri langur,mammalia,primates,cercopithecidae,semnopithecus,johnii
hoolock gibbon,mammalia,primates,hylobatidae,hoolock,hoolock
slender loris,mammalia,primates,lorisidae,loris,lydekkerianus
indian pangolin,mammalia,pholidota,manidae,manis,crassicaudata
indian crested porcupine,mammalia,rodentia,hystricidae,hystrix,indica
indian giant squirrel,mammalia,rodentia,sciuridae,ratufa,indica
indian palm squirrel,mammalia,rodentia,sciuridae,funambulus,palmarum
black-naped hare,mammalia,lagomorpha,leporidae,lepus,nigricollis
indian flying fox,mammalia,chiroptera,pteropodidae,pteropus,medius
indian cobra,reptilia,squamata,elapidae,naja,naja
king cobra,reptilia,squamata,elapidae,ophiophagus,hannah
common krait,reptilia,squamata,elapidae,bungarus,caeruleus
banded krait,reptilia,squamata,elapidae,bungarus,fasciatus
russell's viper,reptilia,squamata,viperidae,daboia,russelii
saw-scaled viper,reptilia,squamata,viperidae,echis,carinatus
bamboo pit viper,reptilia,squamata,viperidae,craspedocephalus,gramineus
indian rock python,reptilia,squamata,pythonidae,python,molurus
burmese python,reptilia,squamata,pythonidae,python,bivittatus
reticulated python,reptilia,squamata,pythonidae,malayopython,reticulatus
indian rat snake,reptilia,squamata,colubridae,ptyas,mucosa
checkered keelback,reptilia,squamata,colubridae,fowlea,piscator
green vine snake,reptilia,squamata,colubridae,ahaetulla,nasuta
indian sand boa,reptilia,squamata,boidae,eryx,johnii
mugger crocodile,reptilia,crocodylia,crocodylidae,crocodylus,palustris
saltwater crocodile,reptilia,crocodylia,crocodylidae,crocodylus,porosus
nile crocodile,reptilia,crocodylia,crocodylidae,crocodylus,niloticus
gharial,reptilia,crocodylia,gavialidae,gavialis,gangeticus
american alligator,reptilia,crocodylia,alligatoridae,alligator,mississippiensis
bengal monitor,reptilia,squamata,varanidae,varanus,bengalensis
water monitor,reptilia,squamata,varanidae,varanus,salvator
komodo dragon,reptilia,squamata,varanidae,varanus,komodoensis
indian star tortoise,reptilia,testudines,testudinidae,geochelone,elegans
indian flapshell turtle,reptilia,testudines,trionychidae,lissemys,punctata
green sea turtle,reptilia,testudines,cheloniidae,chelonia,mydas
olive ridley sea turtle,reptilia,testudines,cheloniidae,lepidochelys,olivacea
oriental garden lizard,reptilia,squamata,agamidae,calotes,versicolor
indian peafowl,aves,galliformes,phasianidae,pavo,cristatus
great indian bustard,aves,otidiformes,otididae,ardeotis,nigriceps
sarus crane,aves,gruiformes,gruidae,antigone,antigone
greater flamingo,aves,phoenicopteriformes,phoenicopteridae,phoenicopterus,roseus
indian vulture,aves,accipitriformes,accipitridae,gyps,indicus
black kite,aves,accipitriformes,accipitridae,milvus,migrans
great hornbill,aves,bucerotiformes,bucerotidae,buceros,bicornis
rose-ringed parakeet,aves,psittaciformes,psittaculidae,psittacula,krameri
common myna,aves,passeriformes,sturnidae,acridotheres,tristis
house crow,aves,passeriformes,corvidae,corvus,splendens
house sparrow,aves,passeriformes,passeridae,passer,domesticus
rock pigeon,aves,columbiformes,columbidae,columba,livia
domestic chicken,aves,galliformes,phasianidae,gallus,gallus domesticus
domestic duck,aves,anseriformes,anatidae,anas,platyrhynchos domesticus
domestic goose,aves,anseriformes,anatidae,anser,anser domesticus
domestic turkey,aves,galliformes,phasianidae,meleagris,gallopavo domesticus
zebu cattle,mammalia,artiodactyla,bovidae,bos,indicus
domestic water buffalo,mammalia,artiodactyla,bovidae,bubalus,bubalis
domestic yak,mammalia,artiodactyla,bovidae,bos,grunniens
domestic donkey,mammalia,perissodactyla,equidae,equus,asinus
domestic pig,mammalia,artiodactyla,suidae,sus,scrofa domesticus
bactrian camel,mammalia,artiodactyla,camelidae,camelus,bactrianus
african forest elephant,mammalia,proboscidea,elephantidae,loxodonta,cyclotis
african savanna elephant,mammalia,proboscidea,elephantidae,loxodonta,africana
okapi,mammalia,artiodactyla,giraffidae,okapia,johnstoni
ring-tailed lemur,mammalia,primates,lemuridae,lemur,catta
verreaux's sifaka,mammalia,primates,indriidae,propithecus,verreauxi
red ruffed lemur,mammalia,primates,lemuridae,varecia,rubra
aye-aye,mammalia,primates,daubentoniidae,daubentonia,madagascariensis
fossa,mammalia,carnivora,eupleridae,cryptoprocta,ferox
meerkat,mammalia,carnivora,herpestidae,suricata,suricatta
african wild dog,mammalia,carnivora,canidae,lycaon,pictus
pangolin,mammalia,pholidota,manidae,smutsia,temminckii
rock hyrax,mammalia,hyracoidea,procaviidae,procavia,capensis
puff adder,reptilia,squamata,viperidae,bitis,arietans
black mamba,reptilia,squamata,elapidae,dendroaspis,polylepis
green mamba,reptilia,squamata,elapidae,dendroaspis,angusticeps
african rock python,reptilia,squamata,pythonidae,python,sebae
boomslang,reptilia,squamata,colubridae,dispholidus,typus
nile monitor,reptilia,squamata,varanidae,varanus,niloticus
panther chameleon,reptilia,squamata,chamaeleonidae,furcifer,pardalis
leopard tortoise,reptilia,testudines,testudinidae,stigmochelys,pardalis
african sacred ibis,aves,pelecaniformes,threskiornithidae,threskiornis,aethiopicus
shoebill,aves,pelecaniformes,balaenicipitidae,balaeniceps,rex
secretarybird,aves,accipitriformes,sagittariidae,sagittarius,serpentarius
wild yak,mammalia,artiodactyla,bovidae,bos,mutus
proboscis monkey,mammalia,primates,cercopithecidae,nasalis,larvatus
chinese pangolin,mammalia,pholidota,manidae,manis,pentadactyla
sunda pangolin,mammalia,pholidota,manidae,manis,javanica
saiga antelope,mammalia,artiodactyla,bovidae,saiga,tatarica
japanese macaque,mammalia,primates,cercopithecidae,macaca,fuscata
sumatran rhinoceros,mammalia,perissodactyla,rhinocerotidae,dicerorhinus,sumatrensis
javan rhinoceros,mammalia,perissodactyla,rhinocerotidae,rhinoceros,sondaicus
bornean orangutan,mammalia,primates,hominidae,pongo,pygmaeus
sumatran orangutan,mammalia,primates,hominidae,pongo,abelii
siberian tiger,mammalia,carnivora,felidae,panthera,tigris altaica
amur leopard,mammalia,carnivora,felidae,panthera,pardus orientalis
slow loris,mammalia,primates,lorisidae,nycticebus,coucang
tarsier,mammalia,primates,tarsiidae,cephalopachus,bancanus
giant salamander,amphibia,caudata,cryptobranchidae,andrias,davidianus
tokay gecko,reptilia,squamata,gekkonidae,gekko,gecko
chinese alligator,reptilia,crocodylia,alligatoridae,alligator,sinensis
red-crowned crane,aves,gruiformes,gruidae,grus,japonensis
mandarin duck,aves,anseriformes,anatidae,aix,galericulata
golden pheasant,aves,galliformes,phasianidae,chrysolophus,pictus
european adder,reptilia,squamata,viperidae,vipera,berus
grass snake,reptilia,squamata,natricidae,natrix,natrix
european pond turtle,reptilia,testudines,emydidae,emys,orbicularis
common lizard,reptilia,squamata,lacertidae,zootoca,vivipara
european hedgehog,mammalia,eulipotyphla,erinaceidae,erinaceus,europaeus
european beaver,mammalia,rodentia,castoridae,castor,fiber
eurasian red squirrel,mammalia,rodentia,sciuridae,sciurus,vulgaris
european rabbit,mammalia,lagomorpha,leporidae,oryctolagus,cuniculus
european hare,mammalia,lagomorpha,leporidae,lepus,europaeus
mouflon,mammalia,artiodactyla,bovidae,ovis,gmelini
white stork,aves,ciconiiformes,ciconiidae,ciconia,ciconia
mute swan,aves,anseriformes,anatidae,cygnus,olor
common cuckoo,aves,cuculiformes,cuculidae,cuculus,canorus
polar bear,mammalia,carnivora,ursidae,ursus,maritimus
arctic fox,mammalia,carnivora,canidae,vulpes,lagopus
arctic hare,mammalia,lagomorpha,leporidae,lepus,arcticus
muskox,mammalia,artiodactyla,bovidae,ovibos,moschatus
western diamondback rattlesnake,reptilia,squamata,viperidae,crotalus,atrox
timber rattlesnake,reptilia,squamata,viperidae,crotalus,horridus
copperhead,reptilia,squamata,viperidae,agkistrodon,contortrix
common garter snake,reptilia,squamata,natricidae,thamnophis,sirtalis
gila monster,reptilia,squamata,helodermatidae,heloderma,suspectum
desert tortoise,reptilia,testudines,testudinidae,gopherus,agassizii
common snapping turtle,reptilia,testudines,chelydridae,chelydra,serpentina
green iguana,reptilia,squamata,iguanidae,iguana,iguana
american bullfrog,amphibia,anura,ranidae,lithobates,catesbeianus
alpaca,mammalia,artiodactyla,camelidae,vicugna,pacos
vicuna,mammalia,artiodactyla,camelidae,vicugna,vicugna
guanaco,mammalia,artiodactyla,camelidae,lama,guanicoe
green anaconda,reptilia,squamata,boidae,eunectes,murinus
boa constrictor,reptilia,squamata,boidae,boa,constrictor
galapagos tortoise,reptilia,testudines,testudinidae,chelonoidis,niger
marine iguana,reptilia,squamata,iguanidae,amblyrhynchus,cristatus
black caiman,reptilia,crocodylia,alligatoridae,melanosuchus,niger
yacare caiman,reptilia,crocodylia,alligatoridae,caiman,yacare
geoffroy's spider monkey,mammalia,primates,atelidae,ateles,geoffroyi
mantled howler monkey,mammalia,primates,atelidae,alouatta,palliata
common squirrel monkey,mammalia,primates,cebidae,saimiri,sciureus
giant otter,mammalia,carnivora,mustelidae,pteronura,brasiliensis
maned wolf,mammalia,carnivora,canidae,chrysocyon,brachyurus
scarlet macaw,aves,psittaciformes,psittacidae,ara,macao
toco toucan,aves,piciformes,ramphastidae,ramphastos,toco
harpy eagle,aves,accipitriformes,accipitridae,harpia,harpyja
blue poison dart frog,amphibia,anura,dendrobatidae,dendrobates,tinctorius
platypus,mammalia,monotremata,ornithorhynchidae,ornithorhynchus,anatinus
tasmanian devil,mammalia,dasyuromorphia,dasyuridae,sarcophilus,harrisii
sugar glider,mammalia,diprotodontia,petauridae,petaurus,breviceps
quokka,mammalia,diprotodontia,macropodidae,setonix,brachyurus
common brushtail possum,mammalia,diprotodontia,phalangeridae,trichosurus,vulpecula
southern cassowary,aves,casuariiformes,casuariidae,casuarius,casuarius
north island brown kiwi,aves,apterygiformes,apterygidae,apteryx,mantelli
laughing kookaburra,aves,coraciiformes,alcedinidae,dacelo,novaeguineae
sulphur-crested cockatoo,aves,psittaciformes,cacatuidae,cacatua,galerita
kea,aves,psittaciformes,strigopidae,nestor,notabilis
coastal taipan,reptilia,squamata,elapidae,oxyuranus,scutellatus
eastern brown snake,reptilia,squamata,elapidae,pseudonaja,textilis
inland taipan,reptilia,squamata,elapidae,oxyuranus,microlepidotus
frilled lizard,reptilia,squamata,agamidae,chlamydosaurus,kingii
thorny devil,reptilia,squamata,agamidae,moloch,horridus
perentie,reptilia,squamata,varanidae,varanus,giganteus
tuatara,reptilia,rhynchocephalia,sphenodontidae,sphenodon,punctatus
emperor penguin,aves,sphenisciformes,spheniscidae,aptenodytes,forsteri
king penguin,aves,sphenisciformes,spheniscidae,aptenodytes,patagonicus
adelie penguin,aves,sphenisciformes,spheniscidae,pygoscelis,adeliae
gentoo penguin,aves,sphenisciformes,spheniscidae,pygoscelis,papua
african penguin,aves,sphenisciformes,spheniscidae,spheniscus,demersus
wandering albatross,aves,procellariiformes,diomedeidae,diomedea,exulans
atlantic puffin,aves,charadriiformes,alcidae,fratercula,arctica
leopard seal,mammalia,carnivora,phocidae,hydrurga,leptonyx
weddell seal,mammalia,carnivora,phocidae,leptonychotes,weddellii
southern elephant seal,mammalia,carnivora,phocidae,mirounga,leonina
harbor seal,mammalia,carnivora,phocidae,phoca,vitulina
gray seal,mammalia,carnivora,phocidae,halichoerus,grypus
walrus,mammalia,carnivora,odobenidae,odobenus,rosmarus
sea otter,mammalia,carnivora,mustelidae,enhydra,lutris
california sea lion,mammalia,carnivora,otariidae,zalophus,californianus
south american sea lion,mammalia,carnivora,otariidae,otaria,flavescens
bottlenose dolphin,mammalia,artiodactyla,delphinidae,tursiops,truncatus
orca,mammalia,artiodactyla,delphinidae,orcinus,orca
humpback whale,mammalia,artiodactyla,balaenopteridae,megaptera,novaeangliae
west indian manatee,mammalia,sirenia,trichechidae,trichechus,manatus
dugong,mammalia,sirenia,dugongidae,dugong,dugon
hawksbill sea turtle,reptilia,testudines,cheloniidae,eretmochelys,imbricata
loggerhead sea turtle,reptilia,testudines,cheloniidae,caretta,caretta
leatherback sea turtle,reptilia,testudines,dermochelyidae,dermochelys,coriacea
"""

path = glob.glob(str(ROOT / "models" / "kaggle" / "**" / "*labels*.txt"), recursive=True)[0]
rows, seen_name, seen_sci = [], set(), set()


def add(common, taxonomy):
    sci = f"{taxonomy[3]} {taxonomy[4]}"
    if common in seen_name or common == "human" or sci == "homo sapiens":
        return
    seen_name.add(common)
    seen_sci.add(sci)
    rows.append({"common": common, "scientific": sci, "taxonomy": ";".join(taxonomy)})


for line in open(path, encoding="utf-8", errors="replace"):
    f = line.strip().split(";")
    if len(f) >= 7 and f[5] and f[1] in ("mammalia", "aves", "reptilia", "amphibia") and "�" not in f[6]:
        add(f[6], f[1:6])
for line in EXTRA.strip().splitlines():
    p = line.split(",")
    add(p[0], p[1:6])

rows.sort(key=lambda r: r["common"])
with open(ROOT / "src" / "species_list.csv", "w", newline="", encoding="utf-8") as out:
    writer = csv.DictWriter(out, fieldnames=["common", "scientific", "taxonomy"])
    writer.writeheader()
    writer.writerows(rows)
counts = {}
for r in rows:
    key = r["taxonomy"].split(";")[0]
    counts[key] = counts.get(key, 0) + 1
print("species:", len(rows), counts)
