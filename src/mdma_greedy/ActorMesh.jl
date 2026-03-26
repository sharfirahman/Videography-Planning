module ActorMesh

using LinearAlgebra

export ActorFace, ActorMeshStruct, build_actor_mesh
export actor_world_vertices,actor_world_face_center,actor_world_normal

struct ActorFace
    name::Symbol  #defines :front, :back, :left, :right, :top, :bottom
    weight::Float64
    normal::Vector{Float64} #Face direction, will be calling it normal as per the convention
    offset::Vector{Float64}  #The center of the actor is [x,y,z]; this calculates the offset from the center of each face
    color::Symbol
    area::Float64
    corner_indices::Vector{Int} #defines the corners of each side of the face
end

struct ActorMeshStruct
    vertices::Vector{Vector{Float64}} #8 corners, which are 3D points
    faces::Vector{ActorFace}
    edges::Vector{Tuple{Int,Int}}
    width::Float64
    depth::Float64
    height::Float64
end

#Constructing the actor mesh with different weight in different sides- Front face has the highest weight

function build_actor_mesh(;
    front_weight::Float64 = 1.0,
    side_weight::Float64 = 0.5,
    top_weight::Float64 = 0.25,
    back_weight::Float64 = 0.2,
    bottom_weight::Float64 = 0.1,
    actor_width::Float64 = 0.5,
    actor_depth::Float64 = 0.3,
    actor_height::Float64 = 0.7)

    d,w,h = actor_depth,actor_width,actor_height

    #Vertices/corners for the local frame

    vertices = Vector{Vector{Float64}}([
        [-d/2, -w/2, -h/2],     #1
        [ d/2, -w/2, -h/2],     #2
        [ d/2,  w/2, -h/2],     #3
        [-d/2,  w/2, -h/2],     #4
        [-d/2, -w/2,  h/2],     #5
        [ d/2, -w/2,  h/2],     #6
        [ d/2,  w/2,  h/2],     #7
        [-d/2,  w/2,  h/2]      #8
        ])


    #Calculating the face area on each side

    front_back_area = w * h
    left_right_area = d * h
    top_bottom_area = d * w


    faces = ActorFace[ ActorFace(:front, front_weight, [1.0,0.0,0.0], [d/2,0.0,0.0], :blue, front_back_area, [2,3,7,6]),
                        ActorFace(:back, back_weight, [-1.0,0.0,0.0], [-d/2,0.0,0.0], :navy, front_back_area, [1,4,8,5]),
                        ActorFace(:left, side_weight, [0.0,1.0,0.0], [0.0,w/2,0.0], :royalblue, left_right_area, [4,3,7,8]),
                        ActorFace(:right, side_weight, [0.0,-1.0,0.0], [0.0,-w/2,0.0], :deepskyblue, left_right_area, [1,2,6,5]),
                        ActorFace(:top, top_weight, [0.0,0.0,1.0], [0.0,0.0,h/2], :lightblue, top_bottom_area, [5,6,7,8]),
                        ActorFace(:bottom, bottom_weight, [0.0,0.0,-1.0], [0.0,0.0,-h/2], :steelblue, top_bottom_area, [1,2,3,4])                
    ]

    edges = Tuple{Int,Int}[
        (1,2),(2,3),(3,4),(4,1),   # bottom half
        (5,6),(6,7),(7,8),(8,5),   # top half
        (1,5),(2,6),(3,7),(4,8),   # vertical pillars
    ]

    return ActorMeshStruct(vertices, faces, edges, w, d, h)

end


#---function designs----#

#We are designing different transformations to reduce the computational load 

#Vertices transformation to world frame

function actor_world_vertices(mesh::ActorMeshStruct, x::Float64, y::Float64,z::Float64, heading::Float64)
    c,s = cos(heading), sin(heading)
    world_vertices = Vector{Vector{Float64}}()

    for v in mesh.vertices
        wx = x + c*v[1] - s*v[2]
        wy = y + s*v[1] + c*v[2]
        wz = z + v[3]
        push!(world_vertices, [wx,wy,wz])
    end
    return world_vertices
end

#Transforming the local actor frame to world frame with the face center


function actor_world_face_center(mesh::ActorMeshStruct, face::ActorFace, x::Float64, y::Float64,z::Float64, heading::Float64)
    c = cos(heading)
    s = sin(heading)
    world_offset_x = x + face.offset[1]*c - face.offset[2]*s
    world_offset_y = y + face.offset[1]*s + face.offset[2]*c
    world_offset_z = z + face.offset[3]
    return [world_offset_x,world_offset_y,world_offset_z]
end


#Actor direction in the world frame 

function actor_world_normal(face::ActorFace, heading::Float64)
    c = cos(heading)
    s = sin(heading)
    world_normal_x = face.normal[1]*c - face.normal[2]*s
    world_normal_y = face.normal[1]*s + face.normal[2]*c
    world_normal_z = face.normal[3]
    return [world_normal_x,world_normal_y,world_normal_z]
end

end #module