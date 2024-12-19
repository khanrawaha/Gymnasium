// import { NextResponse } from "next/dist/server/web/spec-extension/response";

// import Cookies from "cookies";
// import jwt from "jsonwebtoken";

// export default function middleware(req) {
//   const Key = "secretKey";

//   // Instantiate a Cookies object using the req object
//   const cookies = new Cookies(req);

//   // Retrieve the token from the cookie
//   var token_data = cookies.get("myCookieName");

//   // Check if token_data is valid
//   if (!token_data) {
//     // If token_data is invalid or not present, redirect to login
//     return NextResponse.redirect("http://localhost:3000/login");
//   }

//   try {
//     // Verify and decode the token
//     var decoded = jwt.verify(token_data, Key);

//     let url = req.url;

//     // Check if decoded token is empty or if url includes "/dashboard"
//     if (!decoded || (decoded && url.includes("/dashboard"))) {
//       // If decoded token is empty or url includes "/dashboard", redirect to login
//       return NextResponse.redirect("http://localhost:3000/login");
//     }
//   } catch (error) {
//     console.error("JWT verification failed:", error);
//     // If JWT verification fails, redirect to login
//     return NextResponse.redirect("http://localhost:3000/login");
//   }

//   // Return the original request if no redirect is necessary
//   return NextResponse.next();
// }

// import { NextResponse } from "next/dist/server/web/spec-extension/response";
// import Cookies from "cookies";
// import jwt from "jsonwebtoken";

// export default async function middleware(req, res) {
//   const cookies = new Cookies(req, res);
//   const Key = "secretKey";
//   var token_data = cookies.get("myCookieName");

//   console.log(token_data);

// }

import { NextResponse } from "next/server";

export function middleware(request) {
  var cookie = request.cookies.get("myCookieName");

  if (cookie == "undefined") {
    cookie = "false";
  }

  if (request.nextUrl.pathname.includes("/login") && cookie) {
    return NextResponse.rewrite(new URL("/dashboard", request.url));
  }

  if (request.nextUrl.pathname.startsWith("/dashboard") && !cookie) {
    return NextResponse.rewrite(new URL("/login", request.url));
  }
}

export const config = {
  matcher: ["/login/:path*", "/dashboard/:path*"],
};
