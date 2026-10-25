"""Exercise real HTTP requests, bcrypt and JWTs against disposable PostgreSQL."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from unittest.mock import patch
from uuid import UUID, uuid4

import bcrypt
from api_fixture import APITest
import jwt
import sqlalchemy as sa
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import Session

from vf_api.dependencies import get_session
from vf_api.models import User

PREFIX = "/api/v1/auth"
PASSWORD = "a test password with spaces"


class AuthenticationTest(APITest):
    def test_browser_can_read_retry_after(self):
        response = self.client.get("/healthz", headers={"Origin": "http://127.0.0.1:3000"})
        self.assertEqual(response.headers["access-control-allow-origin"], "http://127.0.0.1:3000")
        self.assertIn("Retry-After", response.headers["access-control-expose-headers"])


    def setUp(self):
        self.email = f"owner-{uuid4().hex}@example.com"

    def register(self, email=None, password=PASSWORD):
        return self.client.post(
            PREFIX + "/register",
            json={"email": email or self.email, "password": password},
        )

    def login(self, email=None, password=PASSWORD):
        return self.client.post(
            PREFIX + "/login",
            json={"email": email or self.email, "password": password},
        )

    def account_tokens(self):
        response = self.register()
        self.assertEqual(response.status_code, 201, response.text)
        response = self.login()
        self.assertEqual(response.status_code, 200)
        return response.json()

    def profile(self, token):
        return self.client.get(
            PREFIX + "/me", headers={"Authorization": f"Bearer {token}"}
        )

    def refresh(self, token):
        return self.client.post(PREFIX + "/refresh", json={"refresh_token": token})

    def assert_unauthorized(self, response):
        self.assertEqual(response.status_code, 401, response.text)
        self.assertEqual(response.headers["www-authenticate"], "Bearer")
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.json(), {"detail": "Invalid or expired credentials"})

    def test_register_login_refresh_and_current_user(self):
        response = self.register(email=self.email.upper())
        self.assertEqual(response.status_code, 201, response.text)
        user = response.json()
        self.assertEqual(set(user), {"id", "email", "created_at"})
        self.assertEqual(user["email"], self.email)
        with Session(self.engine) as session:
            stored = session.get(User, UUID(user["id"]))
            self.assertNotEqual(stored.password_hash, PASSWORD)
            self.assertTrue(stored.password_hash.startswith("$2b$12$"))
            self.assertTrue(
                bcrypt.checkpw(PASSWORD.encode(), stored.password_hash.encode())
            )
        response = self.login(email=self.email.upper())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        tokens = response.json()
        self.assertEqual(tokens["token_type"], "bearer")
        self.assertEqual(tokens["expires_in"], 15 * 60)
        self.assertEqual(tokens["refresh_expires_in"], 7 * 86400)
        self.assertEqual(self.profile(tokens["access_token"]).json(), user)
        renewed = self.refresh(tokens["refresh_token"])
        self.assertEqual(renewed.status_code, 200)
        self.assertNotIn("refresh_token", renewed.json())
        self.assertNotEqual(renewed.json()["access_token"], tokens["access_token"])
        self.assertEqual(self.profile(renewed.json()["access_token"]).json(), user)
        # Stateless refresh credentials remain usable until their fixed expiry.
        self.assertEqual(self.refresh(tokens["refresh_token"]).status_code, 200)

    def test_duplicate_email_and_concurrent_registration(self):
        self.assertEqual(self.register().status_code, 201)
        self.assertEqual(self.register(email=self.email.upper()).status_code, 409)
        email = f"race-{uuid4().hex}@example.com"
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(self.register, [email, email.upper()]))
        self.assertEqual(
            sorted(response.status_code for response in responses), [201, 409]
        )
        with Session(self.engine) as session:
            count = session.scalar(
                sa.select(sa.func.count()).select_from(User).where(User.email == email)
            )
            self.assertEqual(count, 1)
        # A failed registration must not poison subsequent request transactions.
        self.assertEqual(self.login(email=email).status_code, 200)

    def test_unknown_account_and_wrong_password_share_response(self):
        missing = self.login()
        self.assertEqual(self.register().status_code, 201)
        wrong = self.login(password="a different wrong password")
        self.assert_unauthorized(missing)
        self.assert_unauthorized(wrong)
        self.assertEqual(missing.json(), wrong.json())
        self.assertNotIn(PASSWORD, wrong.text)

    def test_password_byte_limit_and_unicode(self):
        for password in ("x" * 11, "x" * 73, "é" * 37):
            with self.subTest(length=len(password)):
                response = self.register(password=password)
                self.assertEqual(response.status_code, 422, response.text)
                for error in response.json()["detail"]:
                    self.assertNotIn("input", error)
                    self.assertNotIn("ctx", error)
        response = self.client.post(
            PREFIX + "/register",
            content=json.dumps(
                {"email": self.email, "password": "invalid-surrogate-\ud800"}
            ),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 422, response.text)
        # Both boundary values fit bcrypt's 72-byte input limit.
        for password in ("x" * 72, "é" * 36):
            email = f"boundary-{uuid4().hex}@example.com"
            self.assertEqual(self.register(email, password).status_code, 201)
            self.assertEqual(self.login(email, password).status_code, 200)

    def test_invalid_payloads_do_not_echo_credentials(self):
        sentinel = "private-value-not-for-errors"
        for path, body in (
            ("/register", {"email": "not an email", "password": sentinel}),
            ("/register", {"email": self.email, "password": {"nested": sentinel}}),
            ("/login", {"email": self.email, "password": sentinel, "extra": sentinel}),
            ("/refresh", {"refresh_token": {"nested": sentinel}}),
            ("/refresh", {"refresh_token": sentinel * 200}),
        ):
            with self.subTest(path=path):
                response = self.client.post(PREFIX + path, json=body)
                self.assertEqual(response.status_code, 422)
                self.assertNotIn(sentinel, response.text)
                for error in response.json()["detail"]:
                    self.assertNotIn("input", error)
                self.assertEqual(response.headers["cache-control"], "no-store")
        response = self.client.post(
            PREFIX + "/login",
            content='{"password": "' + sentinel,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertNotIn(sentinel, response.text)

    def test_missing_malformed_and_wrong_kind_credentials(self):
        tokens = self.account_tokens()
        self.assert_unauthorized(self.client.get(PREFIX + "/me"))
        self.assert_unauthorized(
            self.client.get(PREFIX + "/me", headers={"Authorization": "Basic abc"})
        )
        self.assert_unauthorized(self.profile(tokens["refresh_token"]))
        self.assert_unauthorized(self.refresh(tokens["access_token"]))
        for token in ("bad.token.value", "x" * 4097):
            self.assert_unauthorized(self.profile(token))
        self.assert_unauthorized(self.refresh("bad.token.value"))

    def test_token_signature_expiration_and_required_claims(self):
        tokens = self.account_tokens()
        secret = self.settings.jwt_secret.get_secret_value()
        now = datetime.now(timezone.utc)
        for kind, request in (("access", self.profile), ("refresh", self.refresh)):
            claims = jwt.decode(
                tokens[kind + "_token"],
                secret,
                algorithms=["HS256"],
                audience="visionforge-api",
            )
            variants = [
                {**claims, "exp": now - timedelta(seconds=1)},
                {**claims, "nbf": now + timedelta(minutes=5)},
                {**claims, "iat": now + timedelta(minutes=5)},
                {**claims, "iss": "another-service"},
                {**claims, "aud": "another-api"},
                {**claims, "sub": "not-a-uuid"},
                {**claims, "sub": ["not-a-string"]},
            ]
            variants.extend(
                {key: value for key, value in claims.items() if key != missing}
                for missing in ("sub", "exp", "iat", "nbf", "iss", "aud", "type", "jti")
            )
            for payload in variants:
                self.assert_unauthorized(
                    request(jwt.encode(payload, secret, algorithm="HS256"))
                )
            for algorithm, key in (
                ("HS256", "another-secret-" * 5),
                ("HS384", secret),
                ("none", None),
            ):
                self.assert_unauthorized(
                    request(jwt.encode(claims, key, algorithm=algorithm))
                )

    def test_deleted_users_cannot_use_either_token(self):
        tokens = self.account_tokens()
        with Session(self.engine) as session:
            session.execute(sa.delete(User).where(User.email == self.email))
            session.commit()
        self.assert_unauthorized(self.profile(tokens["access_token"]))
        self.assert_unauthorized(self.refresh(tokens["refresh_token"]))

    def test_health_readiness_and_openapi_contract(self):
        self.assertEqual(self.client.get("/healthz").json(), {"status": "ok"})
        self.assertEqual(self.client.get("/readyz").json(), {"status": "ready"})
        self.assertEqual(self.client.get("/docs").status_code, 200)
        schema = self.client.get("/openapi.json").json()
        self.assertEqual(
            schema["paths"][PREFIX + "/me"]["get"]["security"], [{"HTTPBearer": []}]
        )
        self.assertNotIn("password_hash", str(schema))
        self.assertTrue(
            schema["components"]["schemas"]["Credentials"]["properties"]["password"][
                "writeOnly"
            ]
        )

    def test_database_failure_is_generic_and_liveness_still_works(self):
        def unavailable():
            raise OperationalError(
                "private SQL", None, Exception("private database password")
            )
            yield  # Make this a session dependency without touching PostgreSQL.

        self.app.dependency_overrides[get_session] = unavailable
        try:
            response = self.login()
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json(), {"detail": "Database unavailable"})
            self.assertEqual(self.client.get("/readyz").status_code, 503)
            self.assertEqual(self.client.get("/healthz").status_code, 200)
        finally:
            self.app.dependency_overrides.clear()
        # A reachable database without the migrated user table is also unready.
        with patch(
            "sqlalchemy.orm.Session.execute",
            side_effect=ProgrammingError("SELECT", None, Exception("missing table")),
        ):
            self.assertEqual(self.client.get("/readyz").status_code, 503)
