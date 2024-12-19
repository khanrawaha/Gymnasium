import { cookies } from "next/headers";
import { NextRequest, NextResponse } from "next/server";
import jwt from "jsonwebtoken";

const secretKey = "secret";
const key = new TextEncoder().encode(secretKey);

async function encrypt(payload) {
  return await new SignJWT(payload)
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt()
    .setExpirationTime("10 sec from now")
    .sign(key);
}

async function decrypt(input) {
  const { payload } = await jwtVerify(input, key, {
    algorithms: ["HS256"],
  });
  return payload;
}

async function login(formData) {
  // Verify credentials && get the user
  console.log(formData);
  const payload = {
    userId: "123456",
    username: "example_user",
  };

  // Secret key to sign the token
  const secretKey = "your_secret_key";

  // Create the token

  // Create the session
  const expires = new Date(Date.now() + 10 * 1000);
  const token = jwt.sign(payload, secretKey);

  // Save the session in a cookie
  cookies().set("token", token, { expires, httpOnly: true });
}

async function logout() {
  // Destroy the session
  cookies().set("session", "", { expires: new Date(0) });
}

async function getSession() {
  const session = cookies().get("session")?.value;
  if (!session) return null;
  return await decrypt(session);
}

async function updateSession(request) {
  const session = request.cookies.get("session")?.value;
  if (!session) return;

  // Refresh the session so it doesn't expire
  const parsed = await decrypt(session);
  parsed.expires = new Date(Date.now() + 10 * 1000);
  const res = NextResponse.next();
  res.cookies.set({
    name: "session",
    value: await encrypt(parsed),
    httpOnly: true,
    expires: parsed.expires,
  });
  return res;
}

module.exports = {
  encrypt,
  decrypt,
  login,
  logout,
  getSession,
  updateSession,
};
