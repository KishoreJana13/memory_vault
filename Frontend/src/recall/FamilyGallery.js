//familygallery.js
import { Link } from "react-router-dom";
import member1Photo1 from "./profile_memories/monish/pic1.jpg";
// import member2Photo from "./profile_memories/members/member2.jpg";
// import userPhoto from "./profile_memories/members/user.jpg"; 

const members = [
  { id: 1, name: "Jahnavi", src: member1Photo1, link: "/102" },
//   { id: 2, name: "Member 2", src: member2Photo, link: "/member2" },
//   { id: 3, name: "User", src: userPhoto, link: "/user" },
]

function FamilyGallery() {
  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Member Gallery</h1>
      <div className="grid grid-cols-3 gap-4">
        <h1>GOOD</h1>
        {members.map((member) => (
          <Link key={member.id} to={member.link} className="block text-center">
            <img 
              src={member.src} 
              alt={member.name} 
              className="w-32 h-32 object-cover rounded-full mx-auto border-2 border-gray-300"
            />
            <h2 className="text-lg font-semibold mt-2">{member.name}</h2>
          </Link>
        ))}
      </div>
    </div>
  );
}

export default FamilyGallery;
