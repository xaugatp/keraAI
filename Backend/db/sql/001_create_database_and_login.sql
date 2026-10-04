/* =============================================================================
   KeraAI - one-time database setup                       (run by the OWNER in SSMS)

   What this does (all steps are idempotent - safe to re-run):
     1. Creates the databases  KeraAI  (app)  and  KeraAI_Test  (pytest -m mssql).
     2. Creates the SQL login  kera_app  and a matching user in each database with
        db_datareader + db_datawriter + db_ddladmin.
     3. OPTIONAL: grants the same roles to a Windows login (see @WindowsLogin below).

   How to run:
     - Connect to the server in SSMS as a sysadmin (your Windows login is fine).
     - Edit the three variables in the CONFIG block below.
     - Open this file, press F5 (Execute). Read the PRINT messages in the Messages tab.
     - The tables themselves are NOT created here. Alembic creates them:
           cd Backend ; .venv\Scripts\activate ; alembic upgrade head

   ---------------------------------------------------------------------------
   Prerequisites for SQL-login connections (DB_TRUSTED_CONNECTION=false)
   ---------------------------------------------------------------------------
   * A SQL login only works if the server accepts it. In SSMS: right-click the
     server -> Properties -> Security -> choose
         "SQL Server and Windows Authentication mode"
     then RESTART the SQL Server service (SQL Server Configuration Manager ->
     SQL Server Services -> right-click SQL Server (...) -> Restart). Until then the login
     exists but connecting with it fails with "Login failed for user 'kera_app'".
   * If the app cannot reach the server at all (timeouts / "server not found" / a
     host+port in DB_HOST/DB_PORT is refused), open SQL Server Configuration Manager ->
     SQL Server Network Configuration -> Protocols for <instance> -> enable TCP/IP,
     then restart the service. (Plain `localhost` connections from the same machine can
     still succeed over shared memory with TCP/IP off, but anything using host,port or a
     named instance over TCP will not.)

   ---------------------------------------------------------------------------
   About db_ddladmin
   ---------------------------------------------------------------------------
   db_ddladmin is granted so that ONE login can both run Alembic migrations
   (CREATE/ALTER/DROP TABLE) and serve the API. That is a DEV-ONLY convenience.
   In production, split them: a migration login (db_ddladmin / db_owner) used
   only when deploying, and an app login with just db_datareader + db_datawriter.

   ---------------------------------------------------------------------------
   Secrets
   ---------------------------------------------------------------------------
   The password below is a PLACEHOLDER. Never commit a real password to git; put
   the real one only in Backend/.env (DB_PASSWORD=...), which is git-ignored.
   While @SqlLoginPassword still starts with CHANGE_ME the SQL-login section is
   skipped on purpose (so a login with a publicly known password is never created).
   ============================================================================= */

SET NOCOUNT ON;

/* ----------------------------- CONFIG --------------------------------------- */
DECLARE @AppDatabase    SYSNAME       = N'KeraAI';
DECLARE @TestDatabase   SYSNAME       = N'KeraAI_Test';

-- SQL login used by the app when DB_TRUSTED_CONNECTION=false.
DECLARE @SqlLoginName     SYSNAME       = N'kera_app';
DECLARE @SqlLoginPassword NVARCHAR(128) = N'CHANGE_ME_STRONG_PASSWORD';   -- PLACEHOLDER: replace before running

-- OPTIONAL: Windows login to receive the same roles. The owner's Backend/.env
-- currently uses DB_TRUSTED_CONNECTION=true (Windows auth), so this is the login
-- the API actually connects as. Format: DOMAIN\user or MACHINE\user.
-- Set to NULL to skip this section.
DECLARE @WindowsLogin   SYSNAME       = N'CORP\saugat.poudel';
/* ---------------------------------------------------------------------------- */

DECLARE @sql    NVARCHAR(MAX);
DECLARE @dbs    TABLE (name SYSNAME);
DECLARE @db     SYSNAME;

INSERT INTO @dbs (name) VALUES (@AppDatabase), (@TestDatabase);

/* ---- 1. Databases --------------------------------------------------------- */
DECLARE db_cursor CURSOR LOCAL FAST_FORWARD FOR SELECT name FROM @dbs;
OPEN db_cursor;
FETCH NEXT FROM db_cursor INTO @db;
WHILE @@FETCH_STATUS = 0
BEGIN
    IF DB_ID(@db) IS NULL
    BEGIN
        SET @sql = N'CREATE DATABASE ' + QUOTENAME(@db) + N';';
        EXEC (@sql);
        PRINT N'Created database ' + @db;
    END
    ELSE
        PRINT N'Database ' + @db + N' already exists - skipped.';
    FETCH NEXT FROM db_cursor INTO @db;
END
CLOSE db_cursor;
DEALLOCATE db_cursor;

/* ---- 2. SQL login + users + roles ----------------------------------------- */
IF @SqlLoginPassword LIKE N'CHANGE_ME%'
    PRINT N'SKIPPED SQL login section: edit @SqlLoginPassword (still the placeholder) and re-run if you want SQL-login access.';
ELSE
BEGIN
    IF SUSER_ID(@SqlLoginName) IS NULL
    BEGIN
        -- CREATE LOGIN only accepts a literal password, hence dynamic SQL.
        -- CHECK_POLICY = ON enforces the Windows password policy; turn it OFF only
        -- if your dev machine's policy blocks an otherwise-strong password.
        SET @sql = N'CREATE LOGIN ' + QUOTENAME(@SqlLoginName)
                 + N' WITH PASSWORD = N''' + REPLACE(@SqlLoginPassword, N'''', N'''''') + N''''
                 + N', CHECK_POLICY = ON, DEFAULT_DATABASE = ' + QUOTENAME(@AppDatabase) + N';';
        EXEC (@sql);
        PRINT N'Created SQL login ' + @SqlLoginName;
    END
    ELSE
        PRINT N'SQL login ' + @SqlLoginName + N' already exists - skipped (password NOT changed).';

    DECLARE user_cursor CURSOR LOCAL FAST_FORWARD FOR SELECT name FROM @dbs;
    OPEN user_cursor;
    FETCH NEXT FROM user_cursor INTO @db;
    WHILE @@FETCH_STATUS = 0
    BEGIN
        -- USE only lasts for its own batch, so every per-database step is one EXEC
        -- string. ALTER ROLE ... ADD MEMBER is a no-op when already a member.
        SET @sql = N'USE ' + QUOTENAME(@db) + N';
            IF NOT EXISTS (SELECT 1 FROM sys.database_principals WHERE name = @u)
                CREATE USER ' + QUOTENAME(@SqlLoginName) + N' FOR LOGIN ' + QUOTENAME(@SqlLoginName) + N';
            ALTER ROLE db_datareader ADD MEMBER ' + QUOTENAME(@SqlLoginName) + N';
            ALTER ROLE db_datawriter ADD MEMBER ' + QUOTENAME(@SqlLoginName) + N';
            ALTER ROLE db_ddladmin   ADD MEMBER ' + QUOTENAME(@SqlLoginName) + N';';
        EXEC sp_executesql @sql, N'@u SYSNAME', @u = @SqlLoginName;
        PRINT N'Configured user ' + @SqlLoginName + N' in ' + @db;
        FETCH NEXT FROM user_cursor INTO @db;
    END
    CLOSE user_cursor;
    DEALLOCATE user_cursor;
END

/* ---- 3. OPTIONAL: Windows login ------------------------------------------- */
IF @WindowsLogin IS NULL
    PRINT N'Windows login section skipped (@WindowsLogin is NULL).';
ELSE IF IS_SRVROLEMEMBER(N'sysadmin', @WindowsLogin) = 1
    -- A sysadmin is mapped to dbo in every database and already has full rights;
    -- CREATE USER for it would fail ("login already has an account") and is pointless.
    PRINT N'Windows login ' + @WindowsLogin + N' is already a sysadmin - nothing to grant.';
ELSE
BEGIN
    IF SUSER_ID(@WindowsLogin) IS NULL
    BEGIN
        SET @sql = N'CREATE LOGIN ' + QUOTENAME(@WindowsLogin) + N' FROM WINDOWS WITH DEFAULT_DATABASE = '
                 + QUOTENAME(@AppDatabase) + N';';
        EXEC (@sql);
        PRINT N'Created Windows login ' + @WindowsLogin;
    END

    DECLARE win_cursor CURSOR LOCAL FAST_FORWARD FOR SELECT name FROM @dbs;
    OPEN win_cursor;
    FETCH NEXT FROM win_cursor INTO @db;
    WHILE @@FETCH_STATUS = 0
    BEGIN
        -- Match on SID (not name) so an existing user with a different name is reused.
        -- EXEC () cannot call functions such as QUOTENAME inline, so the statements are
        -- first built into a variable (@stmt) and then executed.
        SET @sql = N'USE ' + QUOTENAME(@db) + N';
            DECLARE @member SYSNAME = (SELECT name FROM sys.database_principals WHERE sid = SUSER_SID(@w));
            DECLARE @stmt NVARCHAR(MAX);
            IF @member IS NULL
            BEGIN
                SET @member = @w;
                SET @stmt = N''CREATE USER '' + QUOTENAME(@member) + N'' FOR LOGIN '' + QUOTENAME(@member) + N'';'';
                EXEC (@stmt);
            END
            SET @stmt = N''ALTER ROLE db_datareader ADD MEMBER '' + QUOTENAME(@member) + N'';''
                      + N''ALTER ROLE db_datawriter ADD MEMBER '' + QUOTENAME(@member) + N'';''
                      + N''ALTER ROLE db_ddladmin ADD MEMBER '' + QUOTENAME(@member) + N'';'';
            EXEC (@stmt);';
        EXEC sp_executesql @sql, N'@w SYSNAME', @w = @WindowsLogin;
        PRINT N'Configured Windows user ' + @WindowsLogin + N' in ' + @db;
        FETCH NEXT FROM win_cursor INTO @db;
    END
    CLOSE win_cursor;
    DEALLOCATE win_cursor;
END

PRINT N'Done. Next: set DB_* in Backend/.env, then run  alembic upgrade head.';
