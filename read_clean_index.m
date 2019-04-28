##Octave code to convert the MAT file into two TSVs for work with Python.
## First TSV contains info I want about images (location, classes present)
## Second TSV contains info I want about the classes (names, images present)
##Also converts the objectPresence and objectIsPart fields to sparse arrays.
##Also finds width/height arrays.


index_path='data/ADE20K_2016_07_26/index_ade20k.mat';
data_dir='data/';
if (exist('ade_index'))
  disp('Using in-memory index');
else
  disp('Loading MAT index');
  ade_index = load(index_path);
  ade_index=ade_index.index;
endif
#These are the stored attributes.
## ade_index.filename     ade_index.objectnames           ade_index.typeset           ade_index.wordnet_hypernym
## ade_index.folder       ade_index.objectPresence        ade_index.wordnet_found     ade_index.wordnet_level1
## ade_index.objectcounts ade_index.proportionClassIsPart ade_index.wordnet_frequency ade_index.wordnet_synonyms
## ade_index.objectIsPart ade_index.scene                 ade_index.wordnet_gloss     ade_indexma.wordnet_synset

##e.g. filename for object 3 is
##: ade_index.filename{3}

disp('Processing index');
filename = ade_index.filename;
folder = ade_index.folder;
objectnames = ade_index.objectnames;
objectcounts = ade_index.objectcounts;
objPresence_sparse = sparse(ade_index.objectPresence);
objIsPart_sparse = sparse(ade_index.objectIsPart);
propClassIsPart = ade_index.proportionClassIsPart;
scene = ade_index.scene;

#I want an index showing folder, scene type, object presence and number where the class is present.
#get row/column and count indices
%row is object number, column is image number, value is the count. 
if (exist('pres_i'))
  disp('using existing sparse array');
else
  [pres_i, pres_j, pres_v] = find(objPresence_sparse);
  [part_i, part_j, part_v] = find(objIsPart_sparse);
endif

disp("Starting Object File")
obj_index_filename="index/ADE20K_obj_index_mk2.tsv";
obj_id=fopen(obj_index_filename,"w")  ;
obj_header=strjoin({"index","name","objectCount","proportionClassIsPart","images_present","images_part"},"\t");
fputs(obj_id,obj_header)
fputs(obj_id,"\n");
## Object lookup
# 1. Index number
# 2. Name
# 3. ObjectCount
# 4. Fraction is part
# 5. image_classes transpose
# 6. image_ispart transpose

disp("Starting Obj output");
Nobj = size(objectnames)(2);
for i = 1:Nobj;
  if (mod(i,100)==0)
    disp([i,Nobj])
  endif
  fprintf(obj_id,"%d\t",i);
  fprintf(obj_id,"%s\t",objectnames{i});
  fprintf(obj_id,"%d\t",objectcounts(i));
  fprintf(obj_id,"%d\t",propClassIsPart(i));

  %print out images where this object is present
  msk = (pres_i==i);
  if (sum(msk)>0)
    arr = [pres_j(msk),pres_v(msk)]';
    fputs(obj_id,"[");
    fprintf(obj_id,"(%d,%d), ",arr);
    fputs(obj_id,"]\t");
  else
    fputs(obj_id,"[]\t");
  endif

  ## print out image numbers where it is part
  ## last column so no closing tab char.
  msk = (part_i==i);
  if (sum(msk)>0)
    arr = [part_j(msk),part_v(msk)]';
    fputs(obj_id,"[");
    fprintf(obj_id,"(%d,%d), ",arr);
    fputs(obj_id,"]");
  else
    fputs(obj_id,"[]");
  endif
  fputs(obj_id,"\n");
endfor
fclose(obj_id)

stop_here = true
if (stop_here)
  break
endif


img_index_filename="index/ADE20K_img_index_mk2.tsv";
img_id=fopen(img_index_filename,"w");
img_header=strjoin({"folder","filename","scene","width","height","classes_present","classes_part"},"\t");
fputs(img_id,img_header);
fputs(img_id,"\n");
## Image lookup Header
# 1. folder
# 2. filename
# 3. scene
# 4. image width
# 5. image height
# 6. image_classes
# 7. image_ispart
disp("Starting Image output");
Nimage = size(folder)(2);
for i = 1:Nimage;
  if (mod(i,100)==0)
    disp([i,Nimage])
  endif
  image_path=strjoin({"data",folder{i},filename{i}},"/");
  jpg_info=imfinfo(image_path);
  width=jpg_info.Width;
  height=jpg_info.Height;
  
  fprintf(img_id,"%s\t",folder{i});
  fprintf(img_id,"%s\t",filename{i});
  fprintf(img_id,"%s\t",scene{i});
  fprintf(img_id,"%d\t",width);
  fprintf(img_id,"%d\t",height);

  msk = (pres_j==i);
  if (sum(msk)>0)
    arr = [pres_i(msk),pres_v(msk)]';
    fputs(img_id,"[");
    fprintf(img_id,"(%d,%d), ",arr);
    fputs(img_id,"]\t");
  else
    fputs(img_id,"[]\t");
  endif
  msk = (part_j==i);
  if (sum(msk)>0)  
    arr = [part_i(msk),part_v(msk)]';
    fputs(img_id,"[");
    fprintf(img_id,"(%d, %d), ",arr);
    fputs(img_id,"]");
  else
    fputs(img_id,"[]");
  endif
  fputs(img_id,"\n");
endfor
fclose(img_id);

