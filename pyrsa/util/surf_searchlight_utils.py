import numpy as np

def defineSearchlight_surface(Surf,Mask,kwargs)
# Defines a surface-based searchlight for the input surfaces and functional mask for a subject.
# It is suggested to use the
# rsa_defineSearchlight function instead of calling this function directly,
# since that implements error checking for the inputs.
#
# INPUTS
#   S:  1 (single hemisphere) or 2 (both hemispheres) list of
#       object containing the following properties
#       (following properties are automatically set when using rsa.readSurf
#       function)
#       - white     vertices belonging to the white surface
#       - pial      vertices belonging to the corresponding pial surface
#       - topo      topology for the white/pial surface
#
#   Mask:           object containing functional mask image with the properties
#                   (following properties are automatically set when using
#                   rsa.readMask function)
#       - dim       3-element list with volume dimensions (in voxels)
#       - mat       4x4 affine transformation matrix (numpy array) that aligns mask with
#                   anatomical image
#       - mask      numpyarray with PROD(VOLDEF.voxsize)
#                   elements (logical or numerical) with brain mask
#
# OPTIONS:
#   sphere:         definition of searchlight circle, either one of:
#                   R   scalar R means fixed radius R (in mm)
#                   [R,C] use maximum radius R to find approximately C voxels
#   linedef:        3-element list which defines
#                       - linedef[0] is the number of steps between pial and white surfaces
#                       - linedef[1] is the starting distance from the white surface
#                       - linedef[2] is the ending distance from the pial surface
#   distance:       metric used to calculate distances of voxels from
#                   center voxel
#   progressstep    number of iterations after which progress is reported
#
# OUTPUT:
#       LI:         (nVox) numpy array with linear voxel indices
#       voxmin:     (nVox x 3) Numpy array of Minimal voxel coordinate in x,y,z direction
#       voxmax:     (nVox x 3) Numpy array of maximal voxel coordinate in x,y,z direction
#       voxel:      (nVox x 3) Numpy array of I,J,K voxel coordinates or
#                   (nVox x 1) numpy array with linear voxel indices for the centers of search-lights
#
# EXAMPLE 1:
#   # Define a surface-based searchlight for the left hemisphere only
#   S = rsa.readSurf(['lh.white.surf.gii'],['lh.pial.surf.gii']);
#   M = rsa.readMask('mask.img');
#   L = rsa.defineSearchlight_surface(S,M);
#
# EXAMPLE 2:
#   # Define a surface-based searchlight for both hemispheres, over a
#   80 voxel searchlight with a radius of 20mm
#   white   = ['lh.white.surf.gii', 'rh.white.surf.gii']
#   pial    = ['lh.pial.surf.gii' , 'rh.pial.surf.gii']
#   S       = rsa.readSurf(white,pial)
#   M       = rsa.readMask('mask.img')
#   L       = rsa.defineSearchlight(S,'mask',M,'sphere',[20,80])
#
# 2/2015 - Joern Diedrichsen & Naveed Ejaz (Original Matlab code)
# 9/2020 - Tal Golan (Python adaptation)

## 1. Checking inputs
#       - optional input arguments

Opt['sphere'] = [30,160]
Opt['linedef'] = [5,0,1]
Opt['distancemetric'] = 'geodesic'
Opt['progressstep'] = 100
Opt['writeMask'] = 0 # For debugging
Opt.update(kwargs) # override defaults with kwargs

#       - check if the surface Nodes are legal
numSurf = len(Surf)
numVert = 0
for i in range(len(Surf)):
    Surf[i].nverts  = Surf[i].white.shape[1]
    numVert         = numVert+Surf[i].nverts

    assert [Surf[i].white.shape[0],Surf[i].pial.shape[0],Surf[i].topo.shape[0]]==[3,3,3], 'Coordinates and surfaces should be 3xP (or 3xQ)'
    assert np.array_equal(np.unique(Surf[i].topo),np.arange(Surf[i].nverts)), 'Faces illegal; should contain all indices from 0 to {}'.format(Surf[i].nverts-1)
    assert size(Surf[i].white.shape[1]==Surf[i].pial.shape[1], 'Coordinates C1 and C2 should have same number of elements'

    # TODO - figure out which data type we need here. Matlab source was double
    Surf[i].white  = np.asarray(Surf[i].white)
    Surf[i].pial   = np.asarray(Surf[i].pial)
    Surf[i].topo   = np.asarray(Surf[i].topo)

# TODO: Port mex to Python-integrated C code

# #       - check whether mex function exists (speed optimization)
# mexfunctions = {'surfing_uniqueidxsperrow','surfing_eucldist'};
# for k = 1:numel(mexfunctions)
#     whichmex    = which(mexfunctions{k});
#     [p,n,e]     = fileparts(whichmex);
#     if strcmpi(e,'.m')
#         warning('SURFING:notoptimal',...
#             '#s was not compiled with mex.\nConsider running "mex #s.c" for increased speed.\n',n,n);
#     end;
# end

#       - check mask integrity
if Mask is not None:
    if ~isstruct(Mask) || ~isfield(Mask,'mat') || ~isfield(Mask,'dim'):
        error('mask should be a struct with fields .mat and .dim')

    if isfield(Mask,'mask') and prod(Mask.dim) != numel(Mask.mask):
        error('Volume mask provided, but number of elements does not match mask.dim')

## 2. Getting sphere definition
radius = Opt['sphere'][0]
fixedradius = len(Opt['sphere'])==1
if not fixedradius:
    targetvoxcount = Opt['sphere'][1]

## 3. Generate a single large surface from the component surfaces
white       = []
pial        = []
ca          = []       # Intermediate coordinates
Isurf       = []       # Index into surface
indxoffset  = 0
for i in range(numSurf):
    Surf[i].n2f     = surfing_nodeidxs2faceidxs(Surf[i].topo);
    Surf[i].ca      = (Surf[i].white+Surf[i].pial)/2;          # avg surface
    ca              = [ca,Surf[i].ca];                         # combine both hems
    white           = [white Surf[i].white];
    pial            = [pial Surf[i].pial];
    Isurf           = [Isurf ones(1,Surf[i].nverts)*i];
    indxoffset[i+1] = indxoffset[i]+Surf[i].nverts;

## 4. Projection ontu the linear voxels
# Find coordinates of points on or between the two surfaces.
# Lines between the surfaces are constructed according to LINEDEF.
# ALLCOORDS(I,J,K) is the I-th spatial coordinate (1,2,3)
# for the J-step along the line between the K-th node on surface 1 and 2
allcoords       = surfing_nodeidxs2coords(white,pial,np.arange(numVert),Opt['linedef'])
# Find the voxel indices corresponding to the coordinates of points above
# ALLLINVOXIDXS(I,K) contains the linear index for the voxel
# associated with node I for the K-th step.
alllinvoxidxs   = surfing_coords2linvoxelidxs(allcoords,Mask)
# For each row seperately, duplicates are replaced by zeros.
unqlinvoxidxs   = surfing_uniqueidxsperrow(alllinvoxidxs)
# Now determine the centerindices to be computed
centerindxs     = np.unique(unqlinvoxidxs(~isnan(unqlinvoxidxs) & unqlinvoxidxs~=0))
ncenters        = numel(centerindxs)
for i in range(numSurf):
    Surf[i].unqlinvoxidxs=unqlinvoxidxs(Isurf==i,:);

del alllinvoxidxs white pial unqlinvoxidxs

## 5. Caluculate the exclusive masks (voxels not used for the surface)
inclMask        = Mask;
inclMask.fname  = 'surfMask.nii';
inclMask.data   = zeros(inclMask.dim);
inclMask.data(centerindxs)  = 1;
if Opt['writeMask']:
    spm_write_vol(inclMask,inclMask.data);

exclMask        = Mask;
exclMask.data   = (Mask.mask>0)-(inclMask.data>0)>0;

# parameters in case targetvoxcount is set.
# strategy: use initial radius, then select all voxels with this radius.
# If too few voxels, increase radius, until enough voxels are selected
# now and then we update the value for the initial radius, so that most of the time
# the initial radius is big enough but sometimes not. This is a tradeoff
# between having more nodes initially (with bigger radius; this slows down
# computing the geodesic distances), and increasing the radius sometimes
# which means re-computing geodesic distances
radiusgrow=1.5; # if radius is too small, multiply by this value and try again (see below)
updateradiuscount=floor(log2(ncenters));                                  # } Set how often we update the optimal radius, if targetvoxcount is set.
updateradiusat=repmat(2,1,updateradiuscount-3) .^ (4:updateradiuscount);  # } This is purely for faster execution. Update after 16, 32, ... nodes
updateradiusratio=0.80; # try to find the radius so that in 80# of the case we don't have to increase it. This is an emperical value.
updateradiusmax=1000; # just be be sure the radius does not grow infinitely

# construct mapping from linear to sub indices
centersubs=surfing_inds2subs(Mask.dim,centerindxs);
[centercoords(:,1),centercoords(:,2),centercoords(:,3)]=rsa.util.affine_transform(centersubs(:,1),centersubs(:,2),centersubs(:,3),Mask.mat);

# construct mapping from linear to sub indices for all the voxel in the volume
lin2sub=surfing_inds2subs(Mask.dim,1:prod(Mask.dim));

# Random sequence of center voxels in volume for better estimation
centerorder=randperm(ncenters);

# allocate space for the output
LI=cell(1,ncenters);

# minimum and maximum values for voxel sub indices; or NaN if node is not
# selected
voxmin=NaN(ncenters,3);
voxmax=NaN(ncenters,3);
node=NaN(ncenters,1);
surfindx=NaN(ncenters,1);
depth=NaN(ncenters,1);

# number of voxels OR radius of the searchlights for each center node.
n=NaN(ncenters,1);      # number of voxels
rs=NaN(ncenters,1);     # radius chosen

ascenter=false(ncenters,1); #keep track which voxels were used as center

voxcountsum=0; # to keep track of average number of voxels

if isfield(Mask,'mask'):
    fprintf('Using #d / #d voxels in functional volume mask\n', ncenters, numel(Mask.mask));

tic();
spm_progress_bar('Init',100);
for k in centerorder:          # Center in random sequence
    # Figure out the nearest node for the voxel + cortical depth
    dist=surfing_eucldist(centercoords(k,:)',ca);
    [a,node(k,1)]=min(dist);
    j=Isurf(node(k));                   # Which surface is it on?
    surfindx(k,1)=j;
    node(k)=node(k)-indxoffset(j);     # Correct the node for the offset

    v2=centercoords(k,:)'-Surf(j).white(:,node(k));
    v1=Surf(j).pial(:,node(k))-Surf(j).white(:,node(k));
    depth(k,1)=v1'*v2/(v1'*v1);

    # general while loop;
    # - if number of voxels is not given it exits after one iteration.
    # - if number of voxels is given, then run one iteration, see if we
    #   found enough voxels, and if not we increase the radius and try
    #   again, until we selected enough voxels.
    radiusk=radius;     # set initial radius
    if radius==0
        radiusk=5;
        radius=5;
    end;
    voxcount=0;
    done=false;
    while not done:        #
        # construct a circular ROI around node NODEIDX, and return the indices of
        # the nodes that are near to this node.
        [nodeidxs,dist]=surfing_circleROI(Surf(j).ca,Surf(j).topo,node(k),radiusk,Opt.distancemetric,Surf(j).n2f);

        # find voxel indices of voxels associated with the nodes in the circular ROI

        linvoxidxs=Surf(j).unqlinvoxidxs(nodeidxs,:);
        n2vk=unique(linvoxidxs(linvoxidxs>0));

        voxcount=numel(n2vk); # number of selected voxels
        if fixedradius # select by radius; we're done, exit while-loop

            n(k,1)=voxcount;
            break; # we're done for this node
        else

            # see how many voxels we have
            if voxcount<targetvoxcount
                # not enough voxels; make radius bigger
                radiusk=radiusk*radiusgrow;

                if radiusk>updateradiusmax
                    done=true; # safety mechanism in case we cannot find
                    if Opt.progressstep
                        warning('surfing_voxelselection:note','could not find #d voxels for node #d, ignoring this node', targetvoxcount, nodeidx);
                # we try again with this larger radius in the next iteration
            else
                # we found enough voxels.

                # sort the distance values to find the nodes that are
                # closest
                [foo,sidxs]=sort(dist);

                # linear indices associated with the nearest nodes, sorted
                # by distance
                linvoxidxs=Surf(j).unqlinvoxidxs(nodeidxs(sidxs),:);

                # select approximately targetvoxcount voxels. nrows means
                # here the number of nodes associated with voxels that are
                # selected
                [n2vk,nrows]=surfing_selectkfirstidxs(targetvoxcount,linvoxidxs);

                # update voxelcount
                voxcount=numel(n2vk);
                n(k,1)=voxcount;
                # distance of furthest node on the intermediate surface
                # (We don't take the (additional) distance
                # between a node on the center surface and the center of
                # the voxel into account.)
                rs(k,1)=dist(sidxs(nrows));
                break # we're done for this node


        # store results
    LI{k}=uint32(n2vk(:))';
    ijk=lin2sub(n2vk,:);             # ijk indices (from linear indices)
    voxmin(k,:)=min(ijk,[],1); # minimum voxel indices
    voxmax(k,:)=max(ijk,[],1); # maximum voxel indices
    ascenter(k)=true;


    # optionally show progress
    if Opt.progressstep
        voxcountsum=voxcountsum+voxcount;
        ascentercount=sum(ascenter);
        if ascentercount==1 || mod(ascentercount,abs(Opt.progressstep))==0 || ascentercount==ncenters; # show progress in beginning, every PROGRESSSTEP nodes, and at the end
            if Opt.progressstep<0, clc(); end
            tc=toc();
            eta=(ncenters-ascentercount)/ascentercount*tc;
            if fixedradius
                rtxt=sprintf('r=#.2f, #.1f vox',radius,voxcountsum/ascentercount);
            else
                rtxt=sprintf('r=#.2f, #.1f vox',mean(rs(ascenter)),voxcountsum/ascentercount);
            end
            fprintf('Completed #d / #d nodes (#.1f##), #s, took #d sec, ETA #d sec\n',ascentercount,ncenters,ascentercount/ncenters*100,rtxt,round(tc),round(eta));
            spm_progress_bar('set',ascentercount/ncenters*100);
        end
    end

    # see if we should update the (hopefully close to optimal) radius,
    # in case the target number of voxels is set.
    if ~fixedradius && sum(updateradiusat==ascentercount)
        ascentercount=sum(ascenter);
        radiusidx=round(updateradiusratio*ascentercount);
        if radiusidx>0
            dssort=sort(rs(ascenter));
            radius=dssort(radiusidx); # find the radius in the updateradiusratio*100-th percentile
            if Opt.progressstep
                fprintf('After #d / #d nodes, radius set to #.2f (#d-th percentile)\n', ascentercount, ncenters, radius, round(updateradiusratio*100));
            end
        end
    end
end
radvox=[rs n];

L.LI       = LI';
L.voxmin	= voxmin;
L.voxmax   = voxmax;
L.voxel    = centerindxs;

return L,exclMask